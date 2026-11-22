import hashlib
import json
from pathlib import Path

from ..captioning.generation import generate_records
from ..captioning import profiles as profile_module
from ..captioning.profiles import CUSTOM_PROFILES, STYLE_LABELS, profiles_for_pipeline
from ..captioning.records import write_records
from .ui_feedback import feedback


class CaptionRecordsTpl:
    DESCRIPTION = "Select a Caption Style for a catalogue model, or use custom profiles for full prompt control and multiple candidates per image. Connect Preview, Save Captions or a processing node."
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_pipeline": ("MODEL",), "image_paths": ("LIST",),
            "profile_ids": ("STRING", {"default": "factual,artistic_analysis", "advanced": True,
                "tooltip": "Comma-separated internal profile IDs used when Caption Style is Use custom profiles."}),
        }, "optional": {
            "profiles_file": ("STRING", {"advanced": True, "default": "", "tooltip": "Optional profiles JSON file. Choose a file or inline JSON, not both."}),
            "profiles_json": ("STRING", {"advanced": True, "default": "", "multiline": True, "tooltip": "Editable versioned profiles JSON. Use prompt_mode raw for Florence task tokens or BLIP prefixes. Blank uses built-in profiles."}),
            "pipeline_kwargs": ("STRING", {"advanced": True, "default": "{}", "multiline": True, "tooltip": "Additional JSON generation options; reserved image/prompt inputs cannot be overridden."}),
            "supplied_author": ("STRING", {"advanced": True, "default": "", "tooltip": "Known author metadata, stored separately. Never infer authorship from visual style."}),
            "records_path": ("STRING", {"advanced": True, "default": "", "tooltip": "Optional native JSONL checkpoint for review and later export without inference. Relative paths use the ComfyUI process working directory."}),
            "overwrite_records": ("BOOLEAN", {"advanced": True, "default": False, "tooltip": "Allow replacing an existing checkpoint. Leave disabled to protect saved records."}),
            "caption_style": ([CUSTOM_PROFILES, *STYLE_LABELS.values()], {"default": CUSTOM_PROFILES,
                "tooltip": "Choose a model-specific prompt. Standard BLIP offers Simple Description. Standard Florence offers Simple and Detailed. Some fine-tunes offer Keyword Caption or additional styles."}),
            "prompt_override": ("STRING", {"default": "", "advanced": True, "multiline": True,
                "tooltip": "Optional prompt for a selected Caption Style. Use BLIP prefixes, Florence caption tokens, or instructions according to the model. The exact prompt is shown after a run."}),
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
                 profiles_json="", pipeline_kwargs="{}", supplied_author="", records_path="", overwrite_records=False,
                 caption_style=CUSTOM_PROFILES, prompt_override=""):
        if records_path and Path(records_path).exists() and not overwrite_records:
            raise FileExistsError(f"Protected record output already exists: {records_path}. Use Caption / Image Preview or Dataset Export with records_file to reuse it without inference; choose a new records_path or explicitly enable overwrite_records to regenerate.")
        profiles = profiles_for_pipeline(model_pipeline, caption_style, profile_ids=profile_ids,
                                         profiles_file=profiles_file, profiles_json=profiles_json, prompt_override=prompt_override)
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
        text = result[1] + (f'\nSaved records: {path}' if path else '\nConnect Save Captions for file output.')
        if result[0] and kwargs.get('caption_style', CUSTOM_PROFILES) != CUSTOM_PROFILES:
            candidate = result[0][0]['candidates'][0]
            text += f"\nPrompt used: {candidate['prompt'][:2000] or '(image only)'}"
        return feedback(result, text)
