from .model_loader_tpl import ModelLoaderTpl
from .caption_generator_tpl import CaptionGeneratorTpl


class Florence2NodeTpl:
    def __init__(self):
        self.loader = ModelLoaderTpl()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "florence-community/Florence-2-base"}),  # Editable model name
                "image_paths": ("LIST",),  # List of image paths
                "task_prompt": (["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"], {"default": "<CAPTION>"}),  # Caption Type
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 1024, "min": 32, "max": 2048}),  # Max output length
                "device_mode": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "trust_remote_code": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("generated_captions",)
    FUNCTION = "generate_captions"
    CATEGORY = "Transformers Pipeline"

    def load_model(self, model_name, device_mode="auto", trust_remote_code=False):
        return self.loader.load_model(
            model_name, task="image-text-to-text", device_mode=device_mode,
            trust_remote_code=trust_remote_code,
        )[0]

    def generate_captions(self, model_name, image_paths, task_prompt="<CAPTION>", max_new_tokens=1024,
                          device_mode="auto", trust_remote_code=False):
        try:
            pipe = self.load_model(model_name, device_mode, trust_remote_code)
        except Exception as exc:
            raise RuntimeError(
                f"Florence pipeline could not load {model_name!r}. Native Transformers checkpoints "
                "such as florence-community/Florence-2-base are recommended; legacy Microsoft "
                f"remote-code checkpoints may require a separate compatible environment. {exc}"
            ) from exc
        return CaptionGeneratorTpl().generate_captions(
            pipe, [image_paths], question=task_prompt, max_new_tokens=max_new_tokens,
        )
