import json
from ..captioning.composition import compose_records
from ..captioning.records import CaptionRecord
from .analysis_nodes_tpl import check_destination, record_output


class CaptionComposeTpl:
    DESCRIPTION = 'Build a caption from existing text. Add {labels} for selected model labels. Original captions, scores and review history stay in the records.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'caption_records': ('LIST',),
            'template': ('STRING', {'default': '{caption}', 'label': 'Caption format', 'multiline': True,
                                  'tooltip': 'Use {caption} for caption text alone, or {caption} {labels} to append model labels. For labels with score details use {classifications}. Also available: {tags}, {attributes}, {metadata}, {prefix}.'})},
            'optional': {'selection': ('STRING', {'default': '{}', 'label': 'Evidence options (JSON)', 'multiline': True,
                'tooltip': 'For example, {"min_score": 0.6} includes labels scoring at least 0.6. Scores are model-specific, not probabilities. This keeps every image. top_k limits labels per analysis (default 1); min_score defaults to 0. Optional arrays: profile_ids, candidate_ids, analysis_ids, groups, metadata_fields. Select profile_ids or candidate_ids explicitly to chain derived captions; empty candidate/profile arrays select none. Stale, rejected or failed sources cannot be reused.'}),
                'prefix': ('STRING', {'default': ''}), 'profile_id': ('STRING', {'default': 'composed'}),
                'records_path': ('STRING', {'default': '', 'label': 'Optional records file', 'tooltip': 'Optional new JSONL checkpoint for later review/export; existing files are protected.'})}}
    RETURN_TYPES = ('LIST', 'STRING')
    RETURN_NAMES = ('caption_records', 'status')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Composition'
    def execute(self, caption_records, template, selection='{}', prefix='', profile_id='composed', records_path=''):
        check_destination(records_path)
        return record_output(compose_records([CaptionRecord.from_dict(r) for r in caption_records], template,
            selection=json.loads(selection), prefix=prefix, profile_id=profile_id), records_path)


class CaptionSynthesisTpl:
    DESCRIPTION = 'Optional text-generation synthesis of selected evidence. Appends clean caption text with unreviewed/unverified metadata, shown separately in Preview. Does not verify facts or artist identity; review before training export.'
    @classmethod
    def INPUT_TYPES(cls):
        schema = CaptionComposeTpl.INPUT_TYPES()
        schema['required'] = {'model_pipeline': ('MODEL',), **schema['required']}
        schema['required']['template'][1]['default'] = 'Write a concise description from: {caption} {classifications} {metadata} {prefix}'
        schema['optional']['profile_id'][1]['default'] = 'synthetic'
        schema['optional']['generation_options'] = ('STRING', {'default': '{"max_new_tokens":128,"do_sample":false}', 'multiline': True})
        return schema
    RETURN_TYPES = ('LIST', 'STRING')
    RETURN_NAMES = ('caption_records', 'status')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Composition'
    def execute(self, model_pipeline, caption_records, template, selection='{}', prefix='', profile_id='synthetic', records_path='', generation_options='{}'):
        check_destination(records_path)
        return record_output(compose_records([CaptionRecord.from_dict(r) for r in caption_records], template,
            selection=json.loads(selection), prefix=prefix, profile_id=profile_id, pipe=model_pipeline,
            generation_options=json.loads(generation_options)), records_path)
