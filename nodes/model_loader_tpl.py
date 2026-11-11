# Time to use our the model loader node

import json
from pathlib import Path

import huggingface_hub
import torch
from transformers import pipeline, BitsAndBytesConfig, __version__ as transformers_version
from transformers.pipelines import get_supported_tasks

class ModelLoaderTpl:
    def __init__(self):
        self.model_cache = {}  # One pipeline per loader; replaced when loading options change.

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
                "dtype": (["auto", "float32", "float16", "bfloat16"], {"default": "auto"}),
                "revision": ("STRING", {"default": ""}),
                "local_files_only": ("BOOLEAN", {"default": False}),
                "model_kwargs": ("STRING", {"default": "{}", "multiline": True, "tooltip": "JSON options passed to model loading, e.g. attn_implementation or cache_dir."}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model_pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "Transformers Pipeline"

    def load_model(self, model_name_or_path, task="image-to-text", device_mode="cuda",
                   use_bitsandbytes=False, quantization_type="4bit", trust_remote_code=False,
                   use_fast_tokenizer=True, dtype="auto", revision="", local_files_only=False,
                   model_kwargs="{}"):
        if not model_name_or_path.strip():
            raise ValueError("A Hugging Face model ID or local model path is required")
        supported_tasks = get_supported_tasks()
        if task == "image-to-text" and task not in supported_tasks:
            task = "image-text-to-text"
        if task not in supported_tasks:
            raise ValueError(f"Task {task!r} is unavailable in the installed Transformers version. Check model requirements in an isolated environment.")
        if device_mode not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"Unknown device mode: {device_mode}")
        if device_mode == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        device = "cuda:0" if device_mode != "cpu" and torch.cuda.is_available() else "cpu"
        if dtype not in {"auto", "float32", "float16", "bfloat16"}:
            raise ValueError(f"Unsupported dtype: {dtype}")
        options = json.loads(model_kwargs)
        if not isinstance(options, dict):
            raise ValueError("model_kwargs must be a JSON object")
        reserved = {"device", "device_map", "trust_remote_code", "revision", "dtype", "torch_dtype", "quantization_config", "local_files_only"}
        if reserved.intersection(options):
            raise ValueError("Use the dedicated controls for device, dtype, revision, trust, offline loading and quantization")
        key = (model_name_or_path, task, device, use_bitsandbytes, quantization_type,
               trust_remote_code, use_fast_tokenizer, dtype, revision, local_files_only,
               json.dumps(options, sort_keys=True))
        if key in self.model_cache:
            return (self.model_cache[key],)
        kwargs = dict(task=task, model=model_name_or_path, model_kwargs=options,
                      trust_remote_code=trust_remote_code, use_fast=use_fast_tokenizer,
                      dtype=dtype if dtype == "auto" else getattr(torch, dtype))
        if int(transformers_version.split(".")[0]) >= 5:
            kwargs["local_files_only"] = local_files_only
        else:
            options["local_files_only"] = local_files_only
        if revision:
            kwargs["revision"] = revision
        if use_bitsandbytes:
            if device == "cpu":
                raise ValueError("This node's bitsandbytes mode requires CUDA; disable quantization for CPU")
            if quantization_type not in {"4bit", "8bit"}:
                raise ValueError("quantization_type must be 4bit or 8bit")
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization_type == "4bit", load_in_8bit=quantization_type == "8bit",
                bnb_4bit_compute_dtype=torch.float16,
            )
            kwargs["device_map"] = {"": device}
        else:
            kwargs["device"] = device
        # Do not retain every model ever selected by this node.
        self.model_cache.clear()
        try:
            if local_files_only and not Path(model_name_or_path).is_dir():
                # pipeline() probes config before forwarding model_kwargs on 4.x.
                config_path = huggingface_hub.hf_hub_download(
                    model_name_or_path, "config.json", revision=revision or None,
                    cache_dir=options.get("cache_dir"), local_files_only=True,
                )
                kwargs["model"] = str(Path(config_path).parent)
            loaded = pipeline(**kwargs)
        except Exception as exc:
            raise RuntimeError(f"Could not load {model_name_or_path!r} for {task!r} on {device}: {exc}") from exc
        resolved_revision = getattr(getattr(getattr(loaded, "model", None), "config", None), "_commit_hash", None)
        if resolved_revision is None and Path(kwargs["model"]).parent.name == "snapshots":
            resolved_revision = Path(kwargs["model"]).name
        loaded._tpl_model_info = {"model_id": model_name_or_path, "revision": resolved_revision or revision or None}
        self.model_cache[key] = loaded
        return (loaded,)
