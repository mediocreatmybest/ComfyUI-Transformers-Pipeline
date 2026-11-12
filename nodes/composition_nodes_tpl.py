import json
from ..captioning.composition import compose_records
from ..captioning.records import CaptionRecord
from .analysis_nodes_tpl import check_destination, record_output


class CaptionComposeTpl:
    DESCRIPTION = 'Append a deterministic caption without inference. Existing candidates, edits and approvals are preserved. Classifications remain qualified as possible, with uncalibrated scores.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'caption_records': ('LIST',),
            'template': ('STRING', {'default': '{caption} {classifications} {metadata}', 'multiline': True,
                                  'tooltip': 'Fields: {caption}, {tags}, {attributes}, {classifications}, {metadata}, {prefix}. Only selected evidence is used.'})},
            'optional': {'selection': ('STRING', {'default': '{}', 'multiline': True,
                'tooltip': 'JSON arrays: profile_ids, candidate_ids, analysis_ids, groups, metadata_fields. Optional top_k (default 1) and min_score (default 0). Default {} uses original captions only. Select profile_ids or candidate_ids explicitly to chain derived captions; empty candidate/profile arrays select none. Stale, rejected or failed sources cannot be reused.'}),
                'prefix': ('STRING', {'default': ''}), 'profile_id': ('STRING', {'default': 'composed'}),
                'records_path': ('STRING', {'default': '', 'tooltip': 'Optional new JSONL checkpoint for later review/export; existing files are protected.'})}}
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
