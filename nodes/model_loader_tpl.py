import json

from ..captioning.loading import PipelineCache, legacy_policy

class ModelLoaderTpl:
    DESCRIPTION = "Load a Transformers pipeline. BLIP and Florence community checkpoints work with Desktop Transformers 4.57.3. Qwen3.5 requires the separate 5.22.2 runtime; selecting it does not switch environments."
    def __init__(self):
        self.model_cache = PipelineCache()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name_or_path": ("STRING", {"default": "Salesforce/blip-image-captioning-base", "multiline": False, "tooltip": "Hugging Face model ID or local model directory. Model architecture must be supported by the running Transformers version."}),
                "task": ("STRING", {"default": "image-to-text", "multiline": False, "tooltip": "Transformers pipeline task. BLIP: image-to-text; Florence and conversational VLMs: image-text-to-text. Legacy image-to-text is adapted on Transformers 5."}),
            },
            "optional": {
                "device_mode": (["cuda", "cpu", "auto"], {"default": "cuda", "tooltip": "Legacy device selection, used only when placement is legacy. auto prefers CUDA when available."}),
                "use_bitsandbytes": ("BOOLEAN", {"default": False, "tooltip": "Legacy quantisation switch. Disable it before choosing an explicit quantization mode. Requires bitsandbytes and Accelerate."}),
                "quantization_type": (["4bit", "8bit"], {"default": "8bit", "tooltip": "Legacy bitsandbytes mode: 4bit uses NF4; 8bit uses INT8. Used only with legacy quantization and use_bitsandbytes enabled."}),
                "trust_remote_code": ("BOOLEAN", {"default": False, "tooltip": "Allow model repository Python code to execute. Enable only for trusted repositories that require it; Florence community checkpoints do not need it."}),
                "use_fast_tokenizer": ("BOOLEAN", {"default": True, "tooltip": "Request the fast tokenizer where the model supports one."}),
                "dtype": (["auto", "float32", "float16", "bfloat16"], {"default": "auto", "tooltip": "auto preserves checkpoint precision. Explicit dtype controls native weights or unquantised layers and 4-bit compute. BF16 requires compatible hardware."}),
                "revision": ("STRING", {"default": "", "tooltip": "Optional Hugging Face revision, tag or commit. Blank uses the repository default."}),
                "local_files_only": ("BOOLEAN", {"default": False, "tooltip": "Use cached/local model files only; fail rather than download missing files."}),
                "model_kwargs": ("STRING", {"default": "{}", "multiline": True, "tooltip": "JSON options passed to model loading, e.g. attn_implementation or cache_dir."}),
                "quantization": (["legacy", "native", "int8", "nf4"], {"default": "legacy", "tooltip": "legacy uses the older switches. native preserves checkpoint quantisation/precision. int8 and nf4 request bitsandbytes without full-precision fallback. Existing quantisation is never reapplied."}),
                "placement": (["legacy", "gpu_preferred", "gpu", "offload", "cpu"], {"default": "legacy", "tooltip": "legacy uses device_mode; gpu requires CUDA; gpu_preferred permits CPU for native models if CUDA is unavailable; offload uses Accelerate GPU/CPU placement. No disk offload. NF4 offload and CPU-only bitsandbytes are unsupported."}),
                "gpu_index": ("INT", {"default": 0, "min": 0, "tooltip": "CUDA device index, starting at zero. Used for GPU and GPU/CPU placement."}),
                "gpu_memory": ("STRING", {"default": "", "tooltip": "Weight placement budget, e.g. 8GiB; blank reserves GPU headroom."}),
                "cpu_memory": ("STRING", {"default": "", "tooltip": "Weight placement budget, e.g. 4GiB; blank uses at most half available RAM / 8GiB."}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model_pipeline", "loading_diagnostics")
    OUTPUT_TOOLTIPS = ("Transformers pipeline for captioning nodes.", "JSON diagnostics: runtime, requested and actual precision, quantisation, placement, memory and warnings.")
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
