import hashlib
import json
from pathlib import Path

from ..captioning.generation import generate_records
from ..captioning import profiles as profile_module
from ..captioning.profiles import select_profiles
from ..captioning.records import write_records
from .ui_feedback import feedback


class CaptionRecordsTpl:
    DESCRIPTION = "Generate one candidate per selected profile and image, keeping failures aligned. Built-in IDs: factual, natural_language, subject_tags, artistic_analysis, lora_training. Use Preview or Dataset Export with records_file to reuse saved records without inference."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_pipeline": ("MODEL",), "image_paths": ("LIST",),
            "profile_ids": ("STRING", {"default": "factual,artistic_analysis",
                "tooltip": "Comma-separated profile IDs. Built-in instruction profiles need an instruction-following VLM. For BLIP/Florence, supply raw profiles as shown in the examples."}),
        }, "optional": {
            "profiles_file": ("STRING", {"default": "", "tooltip": "Optional profiles JSON file. Choose a file or inline JSON, not both."}),
            "profiles_json": ("STRING", {"default": "", "multiline": True, "tooltip": "Editable versioned profiles JSON. Use prompt_mode raw for Florence task tokens or BLIP prefixes. Blank uses built-in profiles."}),
            "pipeline_kwargs": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Additional JSON generation options; reserved image/prompt inputs cannot be overridden."}),
            "supplied_author": ("STRING", {"default": "", "tooltip": "Known author metadata, stored separately. Never infer authorship from visual style."}),
            "records_path": ("STRING", {"default": "", "tooltip": "Optional native JSONL checkpoint for review and later export without inference. Relative paths use the ComfyUI process working directory."}),
            "overwrite_records": ("BOOLEAN", {"default": False, "tooltip": "Allow replacing an existing checkpoint. Leave disabled to protect saved records."}),
        }}

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("caption_records", "status")
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "Transformers Pipeline/Datasets"

    @classmethod
    def IS_CHANGED(cls, profiles_file="", profiles_json="", **kwargs):
        if profiles_json:
            return profiles_json
        path = Path(profiles_file) if profiles_file else Path(profile_module.__file__).with_name("profiles.json")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def generate(self, model_pipeline, image_paths, profile_ids="factual,artistic_analysis", profiles_file="",
                 profiles_json="", pipeline_kwargs="{}", supplied_author="", records_path="", overwrite_records=False):
        if records_path and Path(records_path).exists() and not overwrite_records:
            raise FileExistsError(f"Protected record output already exists: {records_path}. Use Caption / Image Preview or Dataset Export with records_file to reuse it without inference; choose a new records_path or explicitly enable overwrite_records to regenerate.")
        if profiles_file and profiles_json:
            raise ValueError("Choose a profiles file or inline profiles JSON")
        profiles = select_profiles(profile_ids, profiles_file or None, json.loads(profiles_json) if profiles_json else None)
        authors = {path: supplied_author for path in image_paths} if supplied_author else None
        records = generate_records(model_pipeline, image_paths, profiles,
                                   pipeline_kwargs=json.loads(pipeline_kwargs), authors=authors)
        if records_path:
            write_records(records, records_path, overwrite_records)
        failures = sum(c.status == "failure" for record in records for c in record.candidates)
        status = f"Images {len(records)}, Candidates {sum(len(r.candidates) for r in records)}, Failures {failures}"
        return ([r.to_dict() for r in records], status)

    def execute(self, **kwargs):
        result = self.generate(**kwargs)
        path = kwargs.get('records_path')
        return feedback(result, result[1] + (f'\nSaved records: {path}' if path else '\nRecords are connected in memory only; set records_path to save for later reuse.'))
