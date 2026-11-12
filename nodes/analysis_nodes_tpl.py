import json
from pathlib import Path
from ..captioning.classification import classify_images, classify_text
from ..captioning.records import CaptionRecord, write_records
from .ui_feedback import feedback


def check_destination(path):
    if path and Path(path).exists():
        raise FileExistsError('Protected records file exists; choose a new path or reuse it with Preview/Export: ' + path)


def record_output(records, path=''):
    if path:
        write_records(records, path)
    failures = sum(a.status == 'failure' for r in records for a in r.analyses)
    status = f'Images {len(records)}, analysis failures {failures}' + (f'\nSaved: {path}' if path else '')
    return feedback(([r.to_dict() for r in records], status), status)


class ZeroShotImageClassificationTpl:
    DESCRIPTION = 'Classify images directly with a zero-shot-image-classification pipeline. One group per instance; clone nodes and share the loader for independent groups. Scores are not calibrated probabilities.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model_pipeline': ('MODEL',), 'image_paths': ('LIST',),
            'candidate_labels': ('STRING', {'default': '["photograph", "painting", "drawing", "illustration"]', 'multiline': True}),
            'group_id': ('STRING', {'default': 'medium'})}, 'optional': {
            'pipeline_kwargs': ('STRING', {'default': '{}', 'multiline': True, 'tooltip': 'Model-supported options, e.g. hypothesis_template. Labels and source use dedicated inputs.'}),
            'records_path': ('STRING', {'default': '', 'tooltip': 'Optional new JSONL checkpoint. Existing files are protected.'})}}
    RETURN_TYPES = ('LIST', 'STRING')
    RETURN_NAMES = ('caption_records', 'status')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Analysis'
    def execute(self, model_pipeline, image_paths, candidate_labels, group_id, pipeline_kwargs='{}', records_path=''):
        check_destination(records_path)
        return record_output(classify_images(model_pipeline, image_paths, json.loads(candidate_labels), group_id,
                                            json.loads(pipeline_kwargs)), records_path)


class ZeroShotTextClassificationTpl:
    DESCRIPTION = 'Classify caption candidates with a zero-shot-classification pipeline, retaining source candidate IDs. Clone nodes for different groups; scores are not calibrated probabilities.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model_pipeline': ('MODEL',), 'caption_records': ('LIST',),
            'candidate_labels': ('STRING', {'default': '["wildlife", "landscape", "architecture"]', 'multiline': True}),
            'group_id': ('STRING', {'default': 'subject'})}, 'optional': {
            'source_profile': ('STRING', {'default': '', 'tooltip': 'Blank classifies all candidates; otherwise choose one source profile.'}),
            'multi_label': ('BOOLEAN', {'default': True, 'tooltip': 'Score labels independently rather than as mutually exclusive alternatives.'}),
            'pipeline_kwargs': ('STRING', {'default': '{}', 'multiline': True}),
            'records_path': ('STRING', {'default': ''})}}
    RETURN_TYPES = ('LIST', 'STRING')
    RETURN_NAMES = ('caption_records', 'status')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Analysis'
    def execute(self, model_pipeline, caption_records, candidate_labels, group_id, source_profile='', multi_label=True, pipeline_kwargs='{}', records_path=''):
        check_destination(records_path)
        options = json.loads(pipeline_kwargs)
        if 'multi_label' in options:
            raise ValueError('Use the multi_label control')
        return record_output(classify_text(model_pipeline, [CaptionRecord.from_dict(r) for r in caption_records],
            json.loads(candidate_labels), group_id, source_profile, {**options, 'multi_label': multi_label}), records_path)
