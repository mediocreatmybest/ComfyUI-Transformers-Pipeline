# Batch Processor to load image list into pipeline in specified batch size
class BatchProcessorTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("LIST",),  # List of image file paths, to match other nodes names
                "batch_size": ("INT", {"default": 1, "min": 1}),  # number of images in each batch for the pipeline
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
