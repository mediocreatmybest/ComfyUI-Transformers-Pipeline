from ..captioning.metadata import metadata_records
from ..captioning.merging import merge_records
from ..captioning.records import CaptionRecord, read_records
from .analysis_nodes_tpl import check_destination, record_output


class ImageMetadataTpl:
    DESCRIPTION = 'Read EXIF into identity-bearing records. Missing EXIF is a valid empty result; unreadable images retain failure records. No inference, no metadata inferred from pixels. IPTC/XMP are not yet read.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'image_paths': ('LIST',)}, 'optional': {'records_path': ('STRING', {'default': ''})}}
    RETURN_TYPES = ('LIST', 'STRING')
    RETURN_NAMES = ('caption_records', 'status')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Analysis'
    def execute(self, image_paths, records_path=''):
        check_destination(records_path)
        return record_output(metadata_records(image_paths), records_path)


class RecordMergeTpl:
    DESCRIPTION = 'Join optional branches by image identity. Exact duplicate IDs are deduplicated; conflicting identities, edits or values are rejected. A saved records_file can be loaded without inference. Chain nodes for more branches.'
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {}, 'optional': {
            'records_a': ('LIST',), 'records_b': ('LIST',), 'records_c': ('LIST',),
            'records_file': ('STRING', {'default': '', 'tooltip': 'Optional saved JSONL branch; can be used alone as a records loader.'}),
            'records_path': ('STRING', {'default': '', 'tooltip': 'Optional new checkpoint. Existing files are protected.'})}}
    RETURN_TYPES = ('LIST', 'STRING')
    RETURN_NAMES = ('caption_records', 'status')
    FUNCTION = 'execute'
    OUTPUT_NODE = True
    CATEGORY = 'Transformers Pipeline/Analysis'
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')
    def execute(self, records_a=None, records_b=None, records_c=None, records_file='', records_path=''):
        check_destination(records_path)
        branches = [[CaptionRecord.from_dict(r) for r in branch or []] for branch in (records_a, records_b, records_c)]
        if records_file:
            branches.append(read_records(records_file))
        return record_output(merge_records(*branches), records_path)
