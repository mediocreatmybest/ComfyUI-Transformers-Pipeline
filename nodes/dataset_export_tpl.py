import json

from ..captioning.export import export_records
from ..captioning.records import CaptionRecord, read_records


class DatasetExportTpl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "destination": ("STRING", {"default": "./caption_exports"}),
            "format": (["txt", "imagefolder", "native"],),
            "profile_id": ("STRING", {"default": "factual"}),
        }, "optional": {
            "caption_records": ("LIST",),
            "records_file": ("STRING", {"default": ""}),
            "caption_type": ("STRING", {"default": ""}),
            "candidate_ids": ("STRING", {"default": "{}", "multiline": True}),
            "approved_only": ("BOOLEAN", {"default": False}),
            "overwrite_existing": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("export_result",)
    FUNCTION = "export"
    OUTPUT_NODE = True
    CATEGORY = "Transformers Pipeline/Datasets"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def export(self, destination, format, profile_id, caption_records=None, records_file="",
               caption_type="", candidate_ids="{}", approved_only=False, overwrite_existing=False):
        if (caption_records is None) == (not records_file):
            raise ValueError("Supply connected caption records or a saved records file, exclusively")
        records = read_records(records_file) if records_file else [CaptionRecord.from_dict(r) for r in caption_records]
        result = export_records(records, destination, format, profile_id=profile_id,
                                caption_type=caption_type or None, candidate_ids=json.loads(candidate_ids),
                                approved_only=approved_only, overwrite=overwrite_existing)
        return (json.dumps(result, ensure_ascii=False),)
