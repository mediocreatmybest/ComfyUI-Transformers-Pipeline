# Caption Generator generating the captions using pipeline
import json

from .pipeline_utils import CAPTION_TASKS, caption_image

class CaptionGeneratorTpl:
    DESCRIPTION = "Generate legacy caption strings in input order, one image at a time. Use Caption / Image Preview with matching image_paths to inspect them, or Caption Records for multiple profiles and per-image failure records."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_pipeline": ("MODEL",),  # the main transformer pipeline from model loader
                "image_batches": ("LIST",),  # List of image in batches
            },
            "optional": {
                "prepend_text": ("STRING", {"default": "", "tooltip": "Text added to each result after inference; not part of the model prompt."}),
                "append_text": ("STRING", {"default": "", "tooltip": "Text appended after inference; not part of the model prompt."}),
                "max_new_tokens": ("INT", {"default": 75, "min": 15, "tooltip": "Maximum generated tokens per image."}),
                "question": ("STRING", {"default": "", "multiline": True, "tooltip": "Model prompt: BLIP prefix, Florence task token such as <CAPTION>, or a question/instruction for a compatible VLM."}),
                "pipeline_kwargs": ("STRING", {"default": "{}", "multiline": True, "tooltip": "Model-specific inference JSON, e.g. {\"enable_thinking\":false,\"generate_kwargs\":{\"do_sample\":false}} for Qwen3.5."}),
            }
        }

    RETURN_TYPES = ("LIST",)  # Output: List of complete generated captions
    RETURN_NAMES = ("captions",)
    FUNCTION = "generate_captions"
    CATEGORY = "Transformers Pipeline"

    def generate_captions(self, model_pipeline, image_batches, question="", prepend_text="", append_text="", max_new_tokens=75, pipeline_kwargs="{}"):
        if model_pipeline is None:
            raise ValueError("No pipeline supplied; check Model Loader")
        if model_pipeline.task not in CAPTION_TASKS:
            raise ValueError(f"Task {model_pipeline.task!r} is not an image captioning task")
        options = json.loads(pipeline_kwargs)
        if not isinstance(options, dict):
            raise ValueError("pipeline_kwargs must be a JSON object")
        reserved = {"images", "image", "text", "question", "prompt", "max_new_tokens", "return_full_text", "return_tensors", "return_type", "skip_special_tokens"}
        if reserved.intersection(options):
            raise ValueError("Use the node inputs for images, prompts and token limit; output format is managed by Caption Generator")
        captions = []
        for batch in image_batches:
            for source in batch:
                try:
                    caption = caption_image(model_pipeline, source, question, max_new_tokens, options)
                except Exception as exc:
                    raise RuntimeError(f"Caption failed for image {len(captions) + 1} ({source}): {exc}") from exc
                captions.append(f"{prepend_text}{caption}{append_text}".strip())
        return (captions,)
