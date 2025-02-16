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

    INPUT_LABELS = {
        "image_paths": "Image Paths",
        "batch_size": "Batch Size",
    }

    RETURN_TYPES = ("LIST",)  # Output: List of image in the batch size
    RETURN_NAMES = ("batches",)
    FUNCTION = "process_batches"
    CATEGORY = "Transformers Pipeline"

    def process_batches(self, image_paths, batch_size=1):
        """
        Splits a list of images into batches for the pipeline.
        """
        try:
            # Create batches of images
            batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
            return (batches,)
        # Throw an error when there is an error...
        except Exception as e:
            print(f"Error processing batches: {e}")
            return ([],)

