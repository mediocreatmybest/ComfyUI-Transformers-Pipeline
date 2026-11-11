# Caption Exporter - Save captions to text file

import os
import re

class CaptionExportTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "captions": ("LIST",),  # List of all generated captions
                "image_paths": ("LIST",),  # List of all image file paths
            },
            "optional": {
                "file_extension": ("STRING", {"default": "txt"}),  # Allow custom file extension
                "keep_extension": ("BOOLEAN", {"default": False}),  # Toggle for filename format
                "overwrite_existing": ("BOOLEAN", {"default": False}),  # Overwrite or skip
                "allow_empty_files": ("BOOLEAN", {"default": False}),  # Do we want to allow empty files or skip
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True # This is an output node, so this will run even if it's not connected to anything.
    FUNCTION = "save_captions"
    CATEGORY = "Transformers Pipeline"

    def save_captions(self, captions, image_paths, file_extension="txt",
                      keep_extension=True, overwrite_existing=False, allow_empty_files=False):
        if len(captions) != len(image_paths):
            raise ValueError(f"Caption/image count mismatch: {len(captions)} captions for {len(image_paths)} images")
        file_extension = file_extension.strip().lower().lstrip(".")
        if not re.fullmatch(r"[a-z0-9]+", file_extension) or file_extension in {
            "png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "exe", "bat", "cmd", "ps1", "sh", "dll", "bin",
        }:
            raise ValueError("Use a plain caption file extension, such as txt")
        destinations = []
        seen = set()
        sources = {os.path.normcase(os.path.realpath(path)) for path in image_paths}
        for caption, path in zip(captions, image_paths):
            if not isinstance(caption, str):
                raise ValueError(f"Caption for {path} must be text")
            destination = (os.fspath(path) if keep_extension else os.path.splitext(path)[0]) + "." + file_extension
            resolved = os.path.normcase(os.path.realpath(destination))
            if resolved in sources or resolved in seen:
                raise ValueError(f"Caption destination collides with an image or another caption: {destination}")
            seen.add(resolved)
            destinations.append((caption.strip(), destination))
        saved = skipped = 0
        for caption, destination in destinations:
            if not caption and not allow_empty_files:
                skipped += 1
                continue
            try:
                # Exclusive creation closes the check-then-write overwrite race.
                with open(destination, "w" if overwrite_existing else "x", encoding="utf-8") as stream:
                    stream.write(caption + "\n")
            except FileExistsError:
                skipped += 1
            else:
                saved += 1
        return (f"Saved {saved}, Skipped {skipped}",)
