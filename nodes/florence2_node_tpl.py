# https://github.com/huggingface/transformers/issues/36106
# Current issue with pipeline and florence2, so skipping pipeline for the moment on this one 
from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image
import os
from .. import be_quiet  # Import global config from root __init__.py

class Florence2NodeTpl:
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None  # Track the currently loaded model

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ("STRING", {"default": "microsoft/Florence-2-large-ft"}),  # Editable model name
                "image_paths": ("LIST",),  # List of image paths
                "task_prompt": (["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"], {"default": "<CAPTION>"}),  # Caption Type
            },
            "optional": {
                "max_new_tokens": ("INT", {"default": 1024, "min": 32, "max": 2048}),  # Max output length
            }
        }

    INPUT_LABELS = {
        "model_name": "Model Name or Path",
        "image_paths": "Image File Paths",
        "task_prompt": "Caption Type",
        "max_new_tokens": "Max New Tokens",
    }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("generated_captions",)
    FUNCTION = "generate_captions"
    CATEGORY = "Transformers Pipeline"

    def load_model(self, model_name):
        """Loads the specified Florence-2 model only once and caches it."""
        if self.model is None or self.processor is None or self.current_model_name != model_name:
            print(f"🔄 Loading Florence-2: {model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16).eval().cuda()
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.current_model_name = model_name  # Track the loaded model

    def generate_captions(self, model_name, image_paths, task_prompt="<CAPTION>", max_new_tokens=1024):
        """Generates captions using Florence-2 in batch mode."""
        try:
            self.load_model(model_name)
            captions = []

            for image_path in image_paths:
                if isinstance(image_path, str) and os.path.exists(image_path):  # Open file path
                    image = Image.open(image_path)
                else:
                    print(f"⚠️ Skipping invalid image path: {image_path}")
                    captions.append("Invalid Image")  # Placeholder if image is missing
                    continue

                # Prepare input for Florence-2
                inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

                # Generate output
                generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"].cuda(),
                    pixel_values=inputs["pixel_values"].cuda(),
                    max_new_tokens=max_new_tokens,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )

                # Decode output
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_caption = self.processor.post_process_generation(
                    generated_text, task=task_prompt, image_size=(image.width, image.height)
                )

                # Extract text from dict
                if isinstance(parsed_caption, dict) and task_prompt in parsed_caption:
                    captions.append(parsed_caption[task_prompt])
                else:
                    captions.append(str(parsed_caption))  # Fallback to string conversion

            return (captions,)

        except Exception as e:
            return ([f"❌ Error generating captions: {e}"],)

