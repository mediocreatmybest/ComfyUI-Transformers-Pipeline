# Batch Processor to load image list into pipeline in specified batch size
class BatchProcessorTpl:
    DESCRIPTION = "Group image paths for Caption Generator. The generator currently captions images sequentially; this does not enable vectorised GPU inference batching."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("LIST",),  # List of image file paths, to match other nodes names
                "batch_size": ("INT", {"default": 1, "min": 1, "tooltip": "Paths per group, not GPU inference batch size. Caption Generator processes each image sequentially."}),
            }
        }

    RETURN_TYPES = ("LIST",)  # Output: List of image in the batch size
    RETURN_NAMES = ("batches",)
    FUNCTION = "process_batches"
    CATEGORY = "Transformers Pipeline"

    def process_batches(self, image_paths, batch_size=1):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return ([image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)],)
