"""Optional LoRA resolution; inference dependencies remain lazy."""
import hashlib
import json
from pathlib import Path
from dataclasses import replace


def local_adapter_fingerprint(use_lora=False, lora_adapter='', adapter_revision=''):
    """Let Comfy's execution cache see local changes before it skips the loader."""
    if not use_lora:
        return None
    path = Path(lora_adapter.strip())
    if not path.is_dir():
        # Also changes the fingerprint when a formerly local folder disappears.
        # The loader supplies the actionable error; no network lookup here.
        return ('not-local', adapter_revision)
    fingerprints = []
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        try:
            # Metadata-only memoisation misses same-size edits with restored mtimes,
            # particularly on Windows. Read only these two adapter files, never base weights.
            with (path/name).open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            fingerprints.append((name, digest))
        except OSError as exc:
            fingerprints.append((name, type(exc).__name__, exc.errno))
    return (adapter_revision, tuple(fingerprints))


def base_identity(value):
    path = Path(value)
    if path.is_dir():
        path = path.resolve()
        if path.parent.name == 'snapshots' and path.parent.parent.name.startswith('models--'):
            return path.parent.parent.name[8:].replace('--', '/'), path.name
        return str(path), None
    return value.rstrip('/'), None


def validate_config(config, base, revision):
    if config.get('peft_type') != 'LORA':
        raise ValueError('This loader supports genuine PEFT LoRA adapters only')
    declared = config.get('base_model_name_or_path')
    if not declared or base_identity(declared)[0] != base_identity(base)[0]:
        raise ValueError('LoRA base model does not match. Select its declared base ID or original cached snapshot; arbitrary copied folders cannot be verified.')
    required_revision = config.get('revision') or base_identity(declared)[1]
    if required_revision and required_revision != revision:
        raise ValueError('LoRA requires a different base revision; select the declared base revision')
    if not config.get('target_modules') or not config.get('r'):
        raise ValueError('LoRA configuration requires target modules and a rank')


def resolve_adapter(policy):
    """Resolve immutable files before cache lookup; never execute repository code."""
    from huggingface_hub import hf_hub_download
    name = policy.lora_adapter.strip()
    local = Path(name).is_dir()
    def fetch(filename, revision):
        try:
            return Path(name)/filename if local else Path(hf_hub_download(name, filename,
                revision=revision or None, cache_dir=policy.model_kwargs.get('cache_dir'),
                token=policy.model_kwargs.get('token'), local_files_only=policy.local_files_only))
        except Exception as exc:
            raise ValueError(f'Cannot read LoRA {filename} at the requested revision: {exc}') from exc
    config_path = fetch('adapter_config.json', policy.adapter_revision)
    try:
        config = json.loads(config_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('LoRA folder must contain a readable adapter_config.json') from exc
    if not isinstance(config, dict):
        raise ValueError('LoRA adapter_config.json must contain a JSON object')
    revision = next((p.name for p in config_path.parents if p.parent.name == 'snapshots'), None)
    if local and policy.adapter_revision and policy.adapter_revision != revision:
        raise ValueError('Adapter revision cannot be verified for this local folder; use its cached snapshot or leave revision blank')
    # Validate identity before downloading weights. Resolve named base revisions through its config.
    from .loading import inspect_model
    _, base_revision, _ = inspect_model(policy)
    checked_config = dict(config)
    if config.get('revision'):
        _, checked_config['revision'], _ = inspect_model(replace(policy,
            model_id=config['base_model_name_or_path'], revision=config['revision']))
    validate_config(checked_config, policy.model_id, base_revision)
    try:
        from peft import PeftConfig
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            PeftConfig.from_peft_type(**config)
        if caught:
            raise ValueError('; '.join(str(w.message) for w in caught))
    except Exception as exc:
        raise ValueError(f'Installed PEFT cannot safely read this LoRA configuration: {exc}') from exc
    weights = fetch('adapter_model.safetensors', revision)
    if not weights.is_file():
        raise ValueError('LoRA requires adapter_model.safetensors; pickle and sharded adapters are not supported')
    digest = hashlib.sha256()
    with weights.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return {'id': name, 'resolved_id': base_identity(str(config_path.parent))[0],
            'revision': revision, 'path': str(config_path.parent),
            'sha256': digest.hexdigest(), 'config': config, 'base_revision': base_revision,
            'reserve_bytes': weights.stat().st_size * 2}


def activate_adapter(pipe, resolved):
    model = pipe.model
    if not callable(getattr(model, 'load_adapter', None)):
        raise ValueError('This model/runtime does not support Transformers LoRA loading')
    result = model.load_adapter(resolved['path'], adapter_name='caption_lora')
    missing = getattr(result, 'missing_keys', [])
    unexpected = getattr(result, 'unexpected_keys', [])
    if unexpected or any('lora_' in key for key in missing):
        raise RuntimeError('LoRA weights did not completely match the model')
    active = model.active_adapters()
    parameters = [(n, p) for n, p in model.named_parameters() if 'lora_' in n]
    layers = [m for m in model.modules() if hasattr(m, 'lora_A')]
    if active != ['caption_lora'] or not parameters or not layers or any(m.disable_adapters for m in layers):
        raise RuntimeError('LoRA activation could not be verified')
    if any(p.device.type == 'meta' for _, p in parameters):
        raise RuntimeError('LoRA weights remain unloaded')
    import torch
    from safetensors.torch import load_file
    from peft import get_peft_model_state_dict
    stored = load_file(str(Path(resolved['path'])/'adapter_model.safetensors'))
    # PEFT wrappers save a prefix absent from Transformers' in-place integration.
    stored = {key.removeprefix('base_model.model.'): value for key, value in stored.items()}
    loaded = get_peft_model_state_dict(model, adapter_name='caption_lora')
    if stored.keys() != loaded.keys() or any(
            not torch.equal(value.cpu(), stored[key].to(dtype=value.dtype)) for key, value in loaded.items()):
        raise RuntimeError(f'Loaded LoRA tensors differ from the saved adapter; refusing partial activation. Saved keys: {list(stored)[:2]}; loaded keys: {list(loaded)[:2]}')
    if not any(bool(torch.count_nonzero(p).item()) for _, p in parameters):
        raise RuntimeError('LoRA contains no nonzero weights')
    return {k: resolved.get(k) for k in ('id', 'resolved_id', 'revision', 'sha256', 'config', 'base_revision')} | {
        'active': True, 'weights_verified': True, 'active_adapters': active, 'parameter_tensors': len(parameters),
        'parameter_bytes': sum(p.numel()*p.element_size() for _, p in parameters)}
