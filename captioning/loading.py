"""Shared loading policy. Importing this module does not import inference libraries."""

from collections import Counter
from dataclasses import asdict, dataclass, field, replace
import gc
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import warnings


def memory_bytes(value):
    if not value:
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(GiB|MiB|GB|MB)", str(value), re.IGNORECASE)
    if not match or float(match[1]) <= 0:
        raise ValueError("Memory budgets must be positive sizes such as 8GiB or 512MiB")
    return int(float(match[1]) * {'gib': 1024**3, 'mib': 1024**2, 'gb': 10**9, 'mb': 10**6}[match[2].lower()])


@dataclass
class LoadingPolicy:
    model_id: str
    task: str = 'image-to-text'
    revision: str | None = None
    backend: str = 'native'
    mode: str = 'native'
    dtype: str = 'auto'
    placement: str = 'gpu_preferred'
    gpu_index: int = 0
    gpu_memory: str = ''
    cpu_memory: str = ''
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    local_files_only: bool = False
    model_kwargs: dict = field(default_factory=dict)
    use_lora: bool = False
    lora_adapter: str = ""
    adapter_revision: str = ""

    def __post_init__(self):
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError('A Hugging Face model ID or local model path is required')
        if self.placement not in {'gpu', 'gpu_preferred', 'offload', 'cpu'}:
            raise ValueError('Unsupported placement; disk offloading is not enabled')
        if self.dtype not in {'auto', 'float32', 'float16', 'bfloat16'}:
            raise ValueError('Unsupported weight dtype')
        if self.backend not in {'native', 'bitsandbytes', 'torchao', 'hqq'}:
            raise ValueError('Unknown quantisation backend')
        if self.backend == 'bitsandbytes' and self.mode not in {'int8', 'nf4'}:
            raise ValueError('bitsandbytes mode must be int8 or nf4')
        if self.backend == 'native' and self.mode != 'native':
            raise ValueError('Native loading cannot request a quantisation mode')
        if type(self.gpu_index) is not int or self.gpu_index < 0:
            raise ValueError('GPU index must be a nonnegative integer')
        if self.use_lora and not self.lora_adapter.strip():
            raise ValueError("Enter a LoRA adapter ID or local folder when Use LoRA is enabled")
        memory_bytes(self.gpu_memory)
        memory_bytes(self.cpu_memory)
        if not isinstance(self.model_kwargs, dict):
            raise ValueError('model_kwargs must be a JSON object')
        reserved = {'device', 'device_map', 'max_memory', 'offload_folder', 'offload_state_dict', 'offload_buffers',
                    'trust_remote_code', 'revision', 'dtype', 'torch_dtype', 'quantization_config',
                    'load_in_8bit', 'load_in_4bit', 'local_files_only', 'config', 'state_dict'}
        if reserved & self.model_kwargs.keys():
            raise ValueError('Use dedicated controls for quantisation, precision, device, memory, revision and trust')
        json.dumps(self.model_kwargs, allow_nan=False)


def legacy_policy(model_id, task='image-to-text', device_mode='cuda', use_bitsandbytes=False,
                  quantization_type='4bit', quantization='legacy', placement='legacy', **kwargs):
    devices = {'cuda': 'gpu', 'cpu': 'cpu', 'auto': 'gpu_preferred'}
    if device_mode not in devices:
        raise ValueError(f'Unknown device mode: {device_mode}')
    if quantization == 'legacy':
        if use_bitsandbytes and quantization_type not in {'8bit', '4bit'}:
            raise ValueError('quantization_type must be 8bit or 4bit')
        quantization = {'8bit': 'int8', '4bit': 'nf4'}[quantization_type] if use_bitsandbytes else 'native'
    elif use_bitsandbytes:
        raise ValueError('Disable legacy use_bitsandbytes when selecting an explicit quantisation policy')
    if quantization not in {'native', 'int8', 'nf4'}:
        raise ValueError('Unsupported quantisation policy')
    return LoadingPolicy(model_id, task=task, backend='native' if quantization == 'native' else 'bitsandbytes',
                         mode=quantization, placement=devices[device_mode] if placement == 'legacy' else placement, **kwargs)


def dependency_versions():
    result = {}
    for name in ('torch', 'transformers', 'accelerate', 'bitsandbytes', 'torchao', 'hqq', 'peft'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def policy_diagnostics(policy):
    result = asdict(policy)
    for key in ('token', 'use_auth_token'):
        if key in result['model_kwargs']:
            result['model_kwargs'][key] = '<redacted>'
    return result


def assess_capability(policy, config, dependencies, cuda_available):
    existing = config.get('quantization_config')
    backend = existing.get('quant_method', 'bitsandbytes') if existing else policy.backend
    status, reason = 'potentially_supported', 'Runtime/model combination requires a successful load and inference test'
    if existing:
        status, reason = 'already_quantised', 'Checkpoint quantisation will be preserved'
    if existing and policy.backend != 'native':
        status, reason = 'unsupported', 'Checkpoint is already quantised; select native to preserve it, without re-quantising'
    elif backend != 'native' and backend in {'bitsandbytes', 'torchao', 'hqq'} and not dependencies.get(backend):
        status, reason = 'missing_optional_dependency', f'Missing optional dependency: {backend}; no full-precision fallback'
    elif not dependencies.get('accelerate'):
        status, reason = 'missing_optional_dependency', 'Accelerate is required for memory planning'
    elif policy.backend in {'torchao', 'hqq'}:
        status, reason = 'unsupported', 'On-the-fly integration is not enabled for this experimental backend'
    elif policy.placement in {'gpu', 'offload'} and not cuda_available:
        status, reason = 'unsupported', 'CUDA requested but unavailable'
    elif backend == 'bitsandbytes' and (policy.placement == 'cpu' or not cuda_available):
        status, reason = 'unsupported', 'CPU-only bitsandbytes is not validated by this loader; select a native CPU checkpoint'
    elif policy.placement == 'offload' and (existing or policy.mode == 'nf4'):
        status, reason = 'unsupported', 'CPU offloading is enabled only for native weights and on-the-fly bitsandbytes INT8'
    return {'status': status, 'reason': reason, 'checkpoint_quantisation': existing, 'backend': backend}


def memory_budgets(policy, *, cpu_available, gpu_free):
    # Keep at least 4 GiB for the OS/other applications; default host weight budget is at most 8 GiB.
    cpu_ceiling = max(0, cpu_available - 4*1024**3)
    cpu = memory_bytes(policy.cpu_memory) or min(8*1024**3, cpu_available // 2, cpu_ceiling)
    if cpu <= 0 or cpu > cpu_ceiling:
        raise ValueError('CPU memory budget would consume reserved system RAM')
    budgets = {}
    if policy.placement in {'cpu', 'offload'} or (policy.placement == 'gpu_preferred' and gpu_free is None):
        budgets['cpu'] = cpu
    if policy.placement != 'cpu' and gpu_free is not None:
        gpu = memory_bytes(policy.gpu_memory) or int(gpu_free * .8)
        if gpu <= 0 or gpu > gpu_free:
            raise ValueError('GPU memory budget exceeds currently free memory')
        budgets = {policy.gpu_index: gpu, **budgets}
    return budgets


def validate_device_map(mapping, policy):
    if not mapping or any(str(device) == 'disk' for device in mapping.values()):
        raise ValueError('Model does not fit the memory budgets without disk offloading; increase budgets or use a smaller model')
    if policy.placement == 'gpu' and any(str(device) == 'cpu' for device in mapping.values()):
        raise ValueError('CPU placement requires an explicit offload policy')
    if policy.placement == 'offload' and all(str(device) == 'cpu' for device in mapping.values()):
        raise ValueError('GPU budget places the entire model on CPU; increase it or explicitly select CPU-only')
    if policy.placement == 'cpu' and any(str(device) != 'cpu' for device in mapping.values()):
        raise ValueError('CPU-only policy cannot place weights on a GPU')


def inspect_model(policy):
    from huggingface_hub import hf_hub_download
    path = Path(policy.model_id)
    subfolder = policy.model_kwargs.get('subfolder', '')
    extra = {key: policy.model_kwargs[key] for key in ('subfolder', 'token') if key in policy.model_kwargs}
    config_path = path / subfolder / 'config.json' if path.is_dir() else Path(hf_hub_download(
        policy.model_id, 'config.json', revision=policy.revision or None,
        cache_dir=policy.model_kwargs.get('cache_dir'), local_files_only=policy.local_files_only, **extra))
    config = json.loads(config_path.read_text(encoding='utf-8'))
    resolved_revision = next((p.name for p in config_path.parents if p.parent.name == 'snapshots'), policy.revision)
    model = str(config_path.parent) if policy.local_files_only or path.is_dir() else policy.model_id
    return model, resolved_revision, config


def checkpoint_quantisation(config):
    backend = config.get('quant_method', 'bitsandbytes')
    mode = (config.get('bnb_4bit_quant_type', 'fp4') if config.get('load_in_4bit') else
            'int8' if config.get('load_in_8bit') else 'checkpoint')
    return backend, mode


def resolve_config(policy, model):
    from transformers import AutoConfig
    config, options = AutoConfig.from_pretrained(model, revision=policy.revision or None,
        trust_remote_code=policy.trust_remote_code, local_files_only=policy.local_files_only,
        return_unused_kwargs=True, **policy.model_kwargs)
    # Hub controls are consumed during config resolution but also needed for weights/processors.
    options.update({k: v for k, v in policy.model_kwargs.items()
                    if k in {'cache_dir', 'token', 'subfolder', 'use_auth_token'}})
    return config, options


def estimate_weight_bytes(model, policy, config_data):
    import torch
    from accelerate.utils import compute_module_sizes
    dtype_name = policy.dtype if policy.dtype != 'auto' else config_data.get('dtype', config_data.get('torch_dtype', 'float32'))
    dtype = getattr(torch, dtype_name or 'float32')
    backend, mode = checkpoint_quantisation(config_data['quantization_config']) if config_data.get('quantization_config') else (policy.backend, policy.mode)
    if backend != 'bitsandbytes':
        return compute_module_sizes(model, dtype=dtype)['']
    from accelerate.utils import CustomDtype
    from transformers.quantizers.base import HfQuantizer
    skips = HfQuantizer.get_modules_to_not_convert(model,
        skip_modules=config_data.get('quantization_config', {}).get('llm_int8_skip_modules'),
        keep_in_fp32_modules=getattr(model, '_keep_in_fp32_modules', None)) if hasattr(model, 'config') else []
    special = {name + '.weight': torch.int8 if mode == 'int8' else CustomDtype.INT4
               for name, module in model.named_modules() if type(module) is torch.nn.Linear
               and not any(name == key or name.startswith(key + '.') or key in name.split('.') for key in skips)}
    # Leave embeddings, output/tied heads, other layer types and buffers at their native size.
    # Reserve quantisation metadata/workspace headroom in addition to the default GPU headroom.
    return int(compute_module_sizes(model, dtype=dtype, special_dtypes=special)[''] * 1.15)


def plan_placement(policy, model, config_data, budgets, *, config=None):
    import torch
    from accelerate import init_empty_weights, infer_auto_device_map
    from transformers.pipelines import check_task

    if config is None:
        config, _ = resolve_config(policy, model)
    _, task_info, _ = check_task(policy.task)
    failures = []
    for auto_class in task_info['pt']:
        try:
            with init_empty_weights():
                empty = auto_class.from_config(config, trust_remote_code=policy.trust_remote_code)
            break
        except (ValueError, TypeError) as exc:
            failures.append(str(exc))
    else:
        raise ValueError('Cannot safely plan memory for this architecture/task: ' + '; '.join(failures))
    empty.tie_weights()
    # Conservative full-size planning also accounts for unquantised CPU modules in INT8 offload.
    dtype_name = policy.dtype if policy.dtype != 'auto' else config_data.get('dtype', config_data.get('torch_dtype', 'float32'))
    dtype = getattr(torch, dtype_name or 'float32')
    if policy.backend != 'native' or config_data.get('quantization_config'):
        dtype = torch.float32
    if policy.placement != 'offload':
        size = estimate_weight_bytes(empty, policy, config.to_dict())
        device = next(iter(budgets))
        if size > budgets[device]:
            raise ValueError(f'Conservative weight estimate {int(size)} bytes exceeds {device} budget {budgets[device]}; use offload or increase the budget')
        return {'': f'cuda:{device}' if isinstance(device, int) else device}
    no_split = empty._get_no_split_modules('auto') if hasattr(empty, '_get_no_split_modules') else empty._no_split_modules
    mapping = infer_auto_device_map(empty, max_memory=budgets, dtype=dtype,
        no_split_module_classes=list(no_split), offload_buffers=False)
    validate_device_map(mapping, policy)
    return mapping


def verify_quantisation(model, backend, mode):
    if backend == 'native':
        return
    if backend == 'bitsandbytes':
        expected = 'Linear8bitLt' if mode == 'int8' else 'Linear4bit'
        modules = [module for module in model.modules() if type(module).__name__ == expected
                   and type(module).__module__.startswith('bitsandbytes')]
        if not modules:
            raise RuntimeError('Requested quantisation was not applied; refusing a full-precision fallback')
        if mode == 'int8' and any(str(module.weight.dtype) != 'torch.int8' for module in modules):
            raise RuntimeError('INT8 modules do not contain packed INT8 weights; refusing a precision fallback')
        if mode in {'nf4', 'fp4'} and any(getattr(module.weight, 'quant_type', None) != mode for module in modules):
            raise RuntimeError('Actual four-bit quantisation type differs from the requested/checkpoint mode')
    elif not getattr(model, 'hf_quantizer', None):
        raise RuntimeError('Checkpoint quantisation was not applied; refusing a full-precision fallback')


def memory_snapshot(gpu_index=0):
    import psutil
    import torch
    result = {'process_rss_bytes': psutil.Process().memory_info().rss,
              'system_available_bytes': psutil.virtual_memory().available}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(gpu_index)
        result.update(gpu_name=torch.cuda.get_device_name(gpu_index), gpu_free_bytes=free, gpu_total_bytes=total,
                      gpu_allocated_bytes=torch.cuda.memory_allocated(gpu_index),
                      gpu_reserved_bytes=torch.cuda.memory_reserved(gpu_index),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu_index))
    return result


def model_diagnostics(pipe):
    model = pipe.model
    parameters = list(model.parameters())
    return {'weight_dtype': str(getattr(model, 'dtype', None)),
            'parameter_dtypes': dict(Counter(str(p.dtype) for p in parameters)),
            'parameter_devices': dict(Counter(str(p.device) for p in parameters)),
            'device_map': {str(k): str(v) for k, v in getattr(model, 'hf_device_map', {'': pipe.device}).items()},
            'quantised_modules': dict(Counter(type(m).__name__ for m in model.modules()
                if type(m).__module__.startswith('bitsandbytes'))),
            'weight_footprint_bytes': model.get_memory_footprint() if hasattr(model, 'get_memory_footprint') else None}


def release_unused_memory():
    gc.collect()
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class PipelineCache:
    """Own at most one pipeline. External/Comfy graph references remain owned by their callers."""
    def __init__(self):
        self.key = None
        self.pipeline = None

    def clear(self):
        self.pipeline = None
        self.key = None
        release_unused_memory()

    def load(self, policy):
        from .adapters import resolve_adapter
        adapter = resolve_adapter(policy) if policy.use_lora else None
        key = json.dumps([asdict(policy), adapter], sort_keys=True)
        if self.pipeline is not None and key == self.key:
            return self.pipeline
        self.clear()
        self.pipeline = load_pipeline(policy, adapter=adapter) if adapter else load_pipeline(policy)
        self.key = key
        return self.pipeline


def load_pipeline(policy, *, adapter=None):
    import torch
    import transformers
    from transformers.pipelines import get_supported_tasks

    from .adapters import resolve_adapter, activate_adapter
    if policy.use_lora and adapter is None:
        adapter = resolve_adapter(policy)
    requested = policy_diagnostics(policy)
    versions = dependency_versions()
    tasks = get_supported_tasks()
    if policy.task == 'image-to-text' and policy.task not in tasks:
        policy = replace(policy, task='image-text-to-text')
    if policy.task not in tasks:
        raise ValueError(f'Task {policy.task!r} is unavailable in Transformers {transformers.__version__}')
    cuda = torch.cuda.is_available()
    if policy.placement in {'gpu', 'offload'} and not cuda:
        raise ValueError('CUDA requested but unavailable')
    if cuda and policy.gpu_index >= torch.cuda.device_count():
        raise ValueError('Requested GPU index is unavailable')
    model, revision, config = inspect_model(policy)
    # Once a snapshot/subfolder resolves to a local directory it is already the model root.
    if Path(model).is_dir() and 'subfolder' in policy.model_kwargs:
        policy = replace(policy, model_kwargs={k: v for k, v in policy.model_kwargs.items() if k != 'subfolder'})
    if revision:
        policy = replace(policy, revision=revision)
    resolved_config, options = resolve_config(policy, model)
    config = resolved_config.to_dict()
    capability = assess_capability(policy, config, versions, cuda)
    if capability['status'] in {'unsupported', 'missing_optional_dependency'}:
        raise ValueError(f"{capability['status']}: {capability['reason']}")
    before = memory_snapshot(policy.gpu_index)
    budgets = memory_budgets(policy, cpu_available=before['system_available_bytes'], gpu_free=before.get('gpu_free_bytes'))
    if adapter:
        # Reserve adapter weights in addition to base weights on every eligible device.
        budgets = {k: v - adapter["reserve_bytes"] for k, v in budgets.items()}
        if min(budgets.values()) <= 0:
            raise ValueError("Memory budget is too small for the LoRA weights")
    mapping = plan_placement(policy, model, config, budgets, config=resolved_config)
    validate_device_map(mapping, policy)
    existing = config.get('quantization_config')
    backend, mode = policy.backend, policy.mode
    if existing:
        from transformers.quantizers import AutoHfQuantizer
        quantizer = AutoHfQuantizer.from_config(existing, pre_quantized=True)
        quantizer.validate_environment(device_map=mapping)
        backend, mode = checkpoint_quantisation(existing)
    elif backend == 'bitsandbytes':
        options['quantization_config'] = transformers.BitsAndBytesConfig(
            load_in_8bit=mode == 'int8', load_in_4bit=mode == 'nf4', bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=getattr(torch, policy.dtype if policy.dtype != 'auto' else 'float16'),
            llm_int8_enable_fp32_cpu_offload=policy.placement == 'offload')
    kwargs = dict(task=policy.task, model=model, config=resolved_config, dtype=policy.dtype,
                  model_kwargs=options, trust_remote_code=policy.trust_remote_code, use_fast=policy.use_fast_tokenizer)
    if policy.revision:
        kwargs['revision'] = policy.revision
    if int(transformers.__version__.split('.')[0]) >= 5:
        kwargs['local_files_only'] = policy.local_files_only
    else:
        options['local_files_only'] = policy.local_files_only
    if backend != 'native' or policy.placement == 'offload':
        kwargs['device_map'] = mapping
        options['offload_state_dict'] = False
    else:
        kwargs['device'] = next(iter(mapping.values()))
    pipe = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            pipe = transformers.pipeline(**kwargs)
        # Transformers 5.22.2 reads this loading flag without consuming it,
        # leaving it in task defaults (and then model.generate). Keep the flag
        # active during loading, but never pass it on to inference.
        for name in ('_preprocess_params', '_forward_params', '_postprocess_params'):
            getattr(pipe, name, {}).pop('local_files_only', None)
        adapter_info = activate_adapter(pipe, adapter) if adapter else {"active": False}
        verify_quantisation(pipe.model, backend, mode)
        actual = model_diagnostics(pipe)
        if 'disk' in actual['device_map'].values():
            raise RuntimeError('Unexpected disk placement from backend')
        actual.update(model_id=policy.model_id, revision=getattr(getattr(pipe.model, 'config', None), '_commit_hash', None) or revision,
                      task=pipe.task, backend=backend, mode=mode, adapter=adapter_info)
        pipe._tpl_model_info = {'model_id': policy.model_id, 'revision': actual['revision'], 'adapter': adapter_info}
        pipe._tpl_loading_diagnostics = {
            'schema_version': 1, 'requested': requested, 'actual': actual, 'runtime': versions,
            'capability': capability, 'verification': {'status': 'verified_supported', 'scope': 'weight_loading_only'},
            'memory_budgets_bytes': {str(k): v for k, v in budgets.items()}, 'memory_before': before,
            'adapter_reserved_bytes_per_device': adapter['reserve_bytes'] if adapter else 0,
            'memory_after': memory_snapshot(policy.gpu_index), 'fallbacks': [],
            'warnings': list(dict.fromkeys(str(w.message) for w in caught)) + (
                ['Adapter does not pin its training-base revision; the selected base revision is recorded.']
                if adapter and not adapter['config'].get('revision') else []) + [
                'Memory budgets constrain conservative weight placement, not activation or temporary allocation peaks.']}
        return pipe
    except Exception as exc:
        pipe = None
        release_unused_memory()
        raise RuntimeError(f'Could not load {policy.model_id!r} with {backend}/{mode}: {exc}; no precision fallback attempted') from exc
