import json

from ..captioning.export import export_records
from ..captioning.records import CaptionRecord, read_records
from .ui_feedback import feedback


class DatasetExportTpl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "destination": ("STRING", {"default": "./caption_exports", "tooltip": "Separate export directory; source images are unchanged. Use a fresh directory for a different dataset selection."}),
            "format": (["txt", "imagefolder", "native"], {"tooltip": "txt and imagefolder export selected training captions. Native v2 is a per-image JSONL archive: selection filters images, retaining ALL their candidates, analyses and review states. Native v1 retains only the selected candidate."}),
            "profile_id": ("STRING", {"default": "factual", "tooltip": "Choose training captions by profile. For native v2 archives, chooses images containing that profile; other profiles are also retained."}),
        }, "optional": {
            "caption_records": ("LIST",),
            "records_file": ("STRING", {"default": "", "tooltip": "Saved native JSONL records, for export without loading a model. Use this or connected caption_records, not both."}),
            "caption_type": ("STRING", {"default": "", "tooltip": "Optional additional caption type filter. Blank selects by profile only."}),
            "candidate_ids": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Optional JSON mapping from image ID to chosen candidate ID, for explicit review selections."}),
            "approved_only": ("BOOLEAN", {"default": False, "tooltip": "Require approval of the selected candidate. Training exports exclude failed/rejected or stale derived captions. Native v2 archives retain all candidates for each qualifying image, including unreviewed/rejected sources."}),
            "overwrite_existing": ("BOOLEAN", {"default": False, "tooltip": "Allow updates to an existing matching export. Disabled by default to protect files."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("export_result",)
    FUNCTION = "execute"
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

    def execute(self, **kwargs):
        try:
            result = self.export(**kwargs)
        except FileExistsError as exc:
            status = {'status': 'protected', 'destination': kwargs['destination'], 'message': str(exc)}
            return feedback((json.dumps(status, ensure_ascii=False),),
                            f'Protected existing export: {exc}\nChoose a new destination or explicitly enable overwrite_existing.')
        report = json.loads(result[0])
        reasons = '\n'.join(str(item['reason'])[:200] for item in report['skipped'][:5])
        return feedback(result, f"Saved {report['exported']}, Skipped {len(report['skipped'])}; format {report['format']}.\n"
                        f"Destination: {report['destination']}\n{report.get('scope', '')}\n{reasons}")
