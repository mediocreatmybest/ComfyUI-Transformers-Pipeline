import json

from ..captioning.loading import PipelineCache, legacy_policy

class ModelLoaderTpl:
    def __init__(self):
        self.model_cache = PipelineCache()

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
                "quantization": (["legacy", "native", "int8", "nf4"], {"default": "legacy"}),
                "placement": (["legacy", "gpu_preferred", "gpu", "offload", "cpu"], {"default": "legacy"}),
                "gpu_index": ("INT", {"default": 0, "min": 0}),
                "gpu_memory": ("STRING", {"default": "", "tooltip": "Weight placement budget, e.g. 8GiB; blank reserves GPU headroom."}),
                "cpu_memory": ("STRING", {"default": "", "tooltip": "Weight placement budget, e.g. 4GiB; blank uses at most half available RAM / 8GiB."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model_pipeline", "loading_diagnostics")
    FUNCTION = "load_model"
    CATEGORY = "Transformers Pipeline"

    def load_model(self, model_name_or_path, task="image-to-text", device_mode="cuda",
                   use_bitsandbytes=False, quantization_type="4bit", trust_remote_code=False,
                   use_fast_tokenizer=True, dtype="auto", revision="", local_files_only=False,
                   model_kwargs="{}", quantization="legacy", placement="legacy", gpu_index=0,
                   gpu_memory="", cpu_memory=""):
        policy = legacy_policy(model_name_or_path, task, device_mode, use_bitsandbytes, quantization_type,
            quantization, placement, dtype=dtype, revision=revision or None, trust_remote_code=trust_remote_code,
            use_fast_tokenizer=use_fast_tokenizer, local_files_only=local_files_only,
            model_kwargs=json.loads(model_kwargs), gpu_index=gpu_index, gpu_memory=gpu_memory, cpu_memory=cpu_memory)
        pipe = self.model_cache.load(policy)
        return (pipe, json.dumps(pipe._tpl_loading_diagnostics, ensure_ascii=False))

    def release_model(self):
        self.model_cache.clear()
