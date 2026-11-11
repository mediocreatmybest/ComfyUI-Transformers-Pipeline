import json

from ..captioning.export import export_records
from ..captioning.records import CaptionRecord, read_records


class DatasetExportTpl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "destination": ("STRING", {"default": "./caption_exports", "tooltip": "Separate export directory; source images are unchanged. Use a fresh directory for a different dataset selection."}),
            "format": (["txt", "imagefolder", "native"], {"tooltip": "txt copies images with UTF-8 sidecars; imagefolder copies images with metadata.jsonl; native preserves rich records in JSONL."}),
            "profile_id": ("STRING", {"default": "factual", "tooltip": "Select the profile ID stored in the caption records."}),
        }, "optional": {
            "caption_records": ("LIST",),
            "records_file": ("STRING", {"default": "", "tooltip": "Saved native JSONL records, for export without loading a model. Use this or connected caption_records, not both."}),
            "caption_type": ("STRING", {"default": "", "tooltip": "Optional additional caption type filter. Blank selects by profile only."}),
            "candidate_ids": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Optional JSON mapping from image ID to chosen candidate ID, for explicit review selections."}),
            "approved_only": ("BOOLEAN", {"default": False, "tooltip": "Export only human-approved candidates. Rejected or failed candidates are excluded from training exports regardless."}),
            "overwrite_existing": ("BOOLEAN", {"default": False, "tooltip": "Allow updates to an existing matching export. Disabled by default to protect files."}),
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
