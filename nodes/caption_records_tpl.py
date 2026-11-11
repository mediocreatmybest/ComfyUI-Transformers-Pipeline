import hashlib
import json
from pathlib import Path

from ..captioning.generation import generate_records
from ..captioning import profiles as profile_module
from ..captioning.profiles import select_profiles
from ..captioning.records import write_records


class CaptionRecordsTpl:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_pipeline": ("MODEL",), "image_paths": ("LIST",),
            "profile_ids": ("STRING", {"default": "factual,artistic_analysis",
                "tooltip": "Comma-separated IDs: factual, natural_language, subject_tags, artistic_analysis, lora_training"}),
        }, "optional": {
            "profiles_file": ("STRING", {"default": ""}),
            "profiles_json": ("STRING", {"default": "", "multiline": True}),
            "pipeline_kwargs": ("STRING", {"default": "{}", "multiline": True}),
            "supplied_author": ("STRING", {"default": ""}),
            "records_path": ("STRING", {"default": ""}),
            "overwrite_records": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("caption_records", "status")
    FUNCTION = "generate"
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
            raise FileExistsError(f"Record output already exists: {records_path}")
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
