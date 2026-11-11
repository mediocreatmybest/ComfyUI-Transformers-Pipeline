# Image Loader to process directory of images into lists
import os
from .ui_feedback import feedback

class ImageLoaderTpl:
    DESCRIPTION = "Discover supported image filenames recursively in sorted order. Discovery does not decode or validate pixels; connect Caption / Image Preview to inspect a small page."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            # As this is a string, it will be displayed as a text box but we can also use another node to select a directory
            "required": {
                "directory_path": ("STRING", {"default": "./images", "multiline": False, "tooltip": "Image folder, searched recursively. Relative paths use ComfyUI's working directory. Supports PNG, JPEG, GIF, BMP, TIFF and WebP; animated files use the first frame during captioning."}),
            },
            # Optional, skip images that already have caption files and don't add to the list
            "optional": {
                "skip_existing_captions": ("BOOLEAN", {"default": False, "tooltip": "Skip images with either name.txt or name.jpg.txt sidecars. Does not inspect separate dataset exports."}),
            }
        }

    RETURN_TYPES = ("LIST", "DICT")  # Outputs: Image list and Image directory mapping if we need it later.
    RETURN_NAMES = ("image_paths", "image_directories")
    FUNCTION = "execute"
    CATEGORY = "Transformers Pipeline"

    def load_images(self, directory_path, skip_existing_captions=False):
        directory_path = os.path.abspath(os.path.expanduser(directory_path))
        if not os.path.isdir(directory_path):
            raise ValueError(f"Image directory does not exist: {directory_path}")
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff', '.webp')
        image_files = []
        image_directories = {}
        discovered = skipped = 0

        def fail(error):
            raise error

        for root, directories, files in os.walk(directory_path, onerror=fail):
            directories.sort()
            for name in sorted(files):
                if not name.lower().endswith(valid_extensions):
                    continue
                path = os.path.join(root, name)
                discovered += 1
                if skip_existing_captions and (
                    os.path.exists(os.path.splitext(path)[0] + ".txt") or os.path.exists(path + ".txt")
                ):
                    skipped += 1
                    continue
                image_files.append(path)
                image_directories[path] = root
        self.discovery_summary = f'Selected {len(image_files)}, Skipped {skipped} existing captions, Discovered {discovered}.\nDirectory: {directory_path}'
        return (image_files, image_directories)

    def execute(self, directory_path, skip_existing_captions=False):
        result = self.load_images(directory_path, skip_existing_captions)
        sample = '\n'.join(os.path.relpath(p, os.path.abspath(os.path.expanduser(directory_path))) for p in result[0][:20])
        message = self.discovery_summary + '\n' + (sample or 'No matching images. Check the folder, supported extensions and skip setting.')
        if len(result[0]) > 20:
            message += '\n… first 20 filenames shown. Use Preview paging for more.'
        return feedback(result, message)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Directory contents and sidecar captions can change between queued runs.
        return float("nan")
