# Image Loader to process directory of images into lists
import os

class ImageLoaderTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # As this is a string, it will be displayed as a text box but we can also use another node to select a directory
            "required": {
                "directory_path": ("STRING", {"default": "./images", "multiline": False}),
            },
            # Optional, skip images that already have caption files and don't add to the list
            "optional": {
                "skip_existing_captions": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LIST", "DICT")  # Outputs: Image list and Image directory mapping if we need it later.
    RETURN_NAMES = ("image_paths", "image_directories")
    FUNCTION = "load_images"
    CATEGORY = "Transformers Pipeline"

    def load_images(self, directory_path, skip_existing_captions=False):
        directory_path = os.path.abspath(os.path.expanduser(directory_path))
        if not os.path.isdir(directory_path):
            raise ValueError(f"Image directory does not exist: {directory_path}")
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp')
        image_files = []
        image_directories = {}

        def fail(error):
            raise error

        for root, directories, files in os.walk(directory_path, onerror=fail):
            directories.sort()
            for name in sorted(files):
                if not name.lower().endswith(valid_extensions):
                    continue
                path = os.path.join(root, name)
                if skip_existing_captions and (
                    os.path.exists(os.path.splitext(path)[0] + ".txt") or os.path.exists(path + ".txt")
                ):
                    continue
                image_files.append(path)
                image_directories[path] = root
        return (image_files, image_directories)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Directory contents and sidecar captions can change between queued runs.
        return float("nan")
