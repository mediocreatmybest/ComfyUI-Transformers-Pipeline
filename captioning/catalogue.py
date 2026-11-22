"""Model choices and preflight guidance shared by the loader and caption nodes."""

from importlib.util import find_spec
import json
from pathlib import Path

from packaging.version import Version


CUSTOM_MODEL = 'Custom model and task'
GROUPS = ('Quick Captions', 'Detailed and Natural Captions', 'Image Classification', 'Text Processing',
          'Advanced Models')
EVIDENCE_LABELS = {
    'desktop_reported': 'Desktop reported',
    'metadata_checked': 'Configuration and processor checked',
    'upstream': 'Upstream documented',
    'experimental': 'Experimental',
    'legacy_incompatible': 'Legacy integration',
}

CATALOGUE = {model['id']: model for model in json.loads(
    Path(__file__).with_name('model_catalogue.json').read_text(encoding='utf-8'))}

_PRESETS = {
    f"{group} · {model['name']}": model
    for group in GROUPS for model in CATALOGUE.values() if model['group'] == group and model['task']
}

_MODEL_TASKS = {'image-to-text', 'image-text-to-text', 'zero-shot-image-classification',
                'text-generation', 'zero-shot-classification'}


def model_choices():
    return [CUSTOM_MODEL, *_PRESETS]


def preset_model(preset):
    if preset == CUSTOM_MODEL:
        return None
    if preset not in _PRESETS:
        raise ValueError(f'Unknown Model Preset: {preset!r}. Select a catalogue model or choose {CUSTOM_MODEL}.')
    return _PRESETS[preset]


def model_description(model):
    name, task = model['name'], model['task']
    detail = f"{name} uses {task}." if task else f"{name} has no native caption preset."
    detail += f" {model['runtime_note']}"
    if model.get('reported_runtime'):
        detail += f" Reported runtime: {model['reported_runtime']}."
    return f"{EVIDENCE_LABELS[model['evidence']]} · {detail}"


def resolve_task(model_id, task, tasks, version, *, trust_remote_code=False, revision=None):
    """Resolve the legacy task alias against the actual registry and validate known combinations."""
    model = CATALOGUE.get(model_id)
    if (model and model['task'] is None
            and revision in (None, '', 'main', 'refs/heads/main', model['audited_revision'])
            and task in {'image-to-text', 'image-text-to-text', 'any-to-any', 'visual-question-answering'}):
        raise ValueError(f"{model['name']} is incompatible with the native Florence image-text-to-text pipeline. "
                         f"The model card uses a legacy processor and explicit remote code. {model['runtime_note']}")
    effective = 'image-text-to-text' if task == 'image-to-text' and task not in tasks and 'image-text-to-text' in tasks else task
    if effective not in tasks:
        suggestion = f" {model_description(model)}" if model else ''
        raise ValueError(f'Unsupported task {task!r} in Transformers {version}.{suggestion} '
                         'Choose a task available in the installed runtime.')
    if not model or model['task'] is None:
        return effective
    recommended = model['task']
    compatible = {recommended, *model.get('additional_tasks', ())}
    if model['prompt_mode'] == 'raw_caption':
        compatible.update(('image-to-text', 'image-text-to-text'))
    if effective in _MODEL_TASKS and effective not in compatible:
        # Other registry tasks remain available in Custom mode and are checked by AutoConfig/AutoModel.
        raise ValueError(f'Unsupported model/task combination for {task!r}. {model_description(model)}')
    if Version(version) < Version(model['minimum_version']):
        raise ValueError(f"Wrong Transformers version {version} for {model['name']}. "
                         f"Use Transformers {model['minimum_version']} or newer. {model['runtime_note']}")
    if model['remote_code'] and not trust_remote_code:
        raise ValueError(f"{model['name']} requires explicitly trusted remote code. {model['runtime_note']}")
    return effective


def check_model_dependencies(model_id):
    model = CATALOGUE.get(model_id, {})
    missing = [name for name in model.get('dependencies', ()) if find_spec(name) is None]
    if missing:
        raise ImportError(f"Missing dependency for {model['name']}: {', '.join(missing)}. "
                          "Install it in the Python environment running ComfyUI.")
