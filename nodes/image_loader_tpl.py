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

    INPUT_LABELS = {
        "directory_path": "Image Directory",
        "skip_existing_captions": "Skip Images with Existing Captions"
    }

    RETURN_TYPES = ("LIST", "DICT")  # Outputs: Image list and Image directory mapping if we need it later.
    RETURN_NAMES = ("image_paths", "image_directories")
    FUNCTION = "load_images"
    CATEGORY = "Transformers Pipeline"

    def load_images(self, directory_path, skip_existing_captions=False):
        """
        Loads image file paths from the given directory (including subdirectories).
        also filters out images that have caption files if selected.
        """
        try:
            valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            image_files = []
            image_directories = {}

            # Let's walk through the directory and find all the images
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        full_path = os.path.join(root, file)

                        # Check if a caption file already exists - Need to allow caption extention to be configurable in the future
                        if skip_existing_captions:
                            base_name = os.path.splitext(full_path)[0]
                            caption_txt = base_name + ".txt"
                            caption_jpg_txt = full_path + ".txt"

                            if os.path.exists(caption_txt) or os.path.exists(caption_jpg_txt):
                                continue  # Skip, whats that skip? Timmy is in the well?!

                        image_files.append(full_path)
                        image_directories[full_path] = root  # Store parent folder for later use

            return (image_files, image_directories)

        except Exception as e:
            print(f"Error loading images: {e}")
            return ([], {})

