# Time to use our the model loader node

import torch
from transformers import pipeline, BitsAndBytesConfig
from .. import be_quiet  # Import as a psudo global config from root __init__.py fix this later

class ModelLoaderTpl:
    def __init__(self):
        self.model_cache = {}  # Store loaded models to prevent additional loading

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name_or_path": ("STRING", {"default": "Salesforce/blip-image-captioning-base", "multiline": False}),
                "task": ("STRING", {"default": "image-to-text", "multiline": False}),
            },
            "optional": {
                "device_mode": (["cuda", "cpu", "auto"], {"default": "cuda"}),  # Device selection
                "use_bitsandbytes": ("BOOLEAN", {"default": False}),  # Enable BitsAndBytes for quant magic
                "quantization_type": (["4bit", "8bit"], {"default": "8bit"}),  # 4-bit or 8-bit options for large models like blip2
                "trust_remote_code": ("BOOLEAN", {"default": False}),  # Enable remote code execution if model requires it (e.g. Florence-2)
                "use_fast_tokenizer": ("BOOLEAN", {"default": True}),  # Use Fast Tokenizer to prevent warnings, whinge whinge...
            }
        }

    INPUT_LABELS = {
        "model_name_or_path": "Model Name or Path",
        "task": "Task Type",
        "device_mode": "Device Mode (CUDA/CPU/Auto)",
        "use_bitsandbytes": "Enable BitsAndBytes Quantisation",
        "quantization_type": "Quantisation Type (4-bit/8-bit)",
        "trust_remote_code": "Trust Remote Code",
        "use_fast_tokenizer": "Use Fast Tokenizer",
    }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model_pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "Transformers Pipeline"

    def load_model(self, model_name_or_path, task="image-to-text", device_mode="cuda",
                   use_bitsandbytes=False, quantization_type="4bit", trust_remote_code=False,
                   use_fast_tokenizer=True):
        """
        Loads a model for Transformers a task.
        added BitsAndBytes quant magic and now with added additional remote execution! Thanks Demtel!.
        """
        try:
            # If BitsAndBytes is enabled, configure the quantmagic
            if use_bitsandbytes:
                if not be_quiet:
                    print(f"⚡ Loading {model_name_or_path} with {quantization_type} quantisation...")
                # Now required for BitsAndBytes when passing 4-bit or 8-bit
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=(quantization_type == "4bit"),
                    load_in_8bit=(quantization_type == "8bit"),
                    bnb_4bit_compute_dtype=torch.float16,
                )

                pipeline_kwargs = {
                    "task": task,
                    "model": model_name_or_path,
                    "model_kwargs": {
                        "quantization_config": quant_config,
                        "low_cpu_mem_usage": True,
                    }
                }

            else:
                # Set device mode (Auto, CUDA, or CPU)
                if device_mode == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                elif device_mode == "cuda" and torch.cuda.is_available():
                    device = 0
                else:
                    device = "cpu"

                if not be_quiet:
                    print(f"🛠 Loading Model: {model_name_or_path} on {device}")

                pipeline_kwargs = {
                    "task": task,
                    "model": model_name_or_path,
                    "device": device,
                    "use_fast": use_fast_tokenizer,  # Enable fast tokenizer to prevent warnings
                }

                # Only add trust_remote_code if it's explicitly enabled
                if trust_remote_code:
                    pipeline_kwargs["trust_remote_code"] = True

            # Prevent redundant model loading
            if model_name_or_path in self.model_cache:
                if not be_quiet:
                    print(f"Using Cached Model: {model_name_or_path}")
                return (self.model_cache[model_name_or_path],)

            # Load pipeline and pass the kwargs that we have collected along the way
            model_pipeline = pipeline(**pipeline_kwargs)

            # Cache the model
            self.model_cache[model_name_or_path] = model_pipeline

            if not be_quiet:
                print(f"Model Successfully Loaded: {model_name_or_path}")

            return (model_pipeline,)

        except Exception as e:
            print(f"Error loading model {model_name_or_path}: {e}")
            return (None,)
