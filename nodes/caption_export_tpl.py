# Caption Exporter - Save captions to text file

import os
import re
from .ui_feedback import feedback

class CaptionExportTpl:
    DESCRIPTION = "Legacy sidecars beside source images. Existing files and empty captions are skipped by default. Use Dataset Export for a separate output directory."
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
                "file_extension": ("STRING", {"default": "txt", "tooltip": "Caption extension only, normally txt. Sidecars are written beside the source images."}),
                "keep_extension": ("BOOLEAN", {"default": False, "tooltip": "False: image.txt. True: image.jpg.txt, avoiding collisions across source extensions."}),
                "overwrite_existing": ("BOOLEAN", {"default": False, "tooltip": "False skips existing sidecars safely. True explicitly replaces them."}),
                "allow_empty_files": ("BOOLEAN", {"default": False, "tooltip": "False skips empty captions. Caption/image counts must always match."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    OUTPUT_NODE = True # This is an output node, so this will run even if it's not connected to anything.
    FUNCTION = "execute"
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

    def execute(self, **kwargs):
        result = self.save_captions(**kwargs)
        return feedback(result, result[0] + '\nDestination: sidecars beside source images. Skips mean existing-file protection or empty captions, not inference failures.')
