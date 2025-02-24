# Caption Exporter - Save captions to text file

import os
from .. import be_quiet  # Import global quiet mode setting

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
        """
        Saves captions as text files next to their paired image.
        Skips empty captions set as default unless 'allow_empty_files' is true.
        Allows custom file extensions, and will defaults to .txt.
        """
        try:
            if not captions:
                if not be_quiet:
                    print("No captions, skipping...")
                return ("No captions, skipping...",)

            saved_count = 0
            skipped_count = 0

            # Check that extension is not something odd
            file_extension = file_extension.strip().lower().lstrip(".")
            if not file_extension or any(ext in file_extension for ext in ["bin", "exe", "bat", "sh", "dll"]):
                file_extension = "txt"  # Default to .txt

            # Skip saving if caption is empty and allow_empty_files is False
            for caption, image_path in zip(captions, image_paths):
                if not caption.strip() and not allow_empty_files:
                    skipped_count += 1
                    continue

                # Extract filename and use the set extension
                base_name = os.path.basename(image_path)
                file_name = f"{base_name}.{file_extension}" if keep_extension else f"{os.path.splitext(base_name)[0]}.{file_extension}"
                caption_path = os.path.join(os.path.dirname(image_path), file_name)

                # Skip if file exists and overwrite is disabled
                if os.path.exists(caption_path) and not overwrite_existing:
                    skipped_count += 1
                    continue

                # Save caption file in UTF-8 for special characters, emojis, etc.
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption.strip() + "\n")
                    saved_count += 1

            if not be_quiet:
                 # Print and Return the status message
                 # Try to move this into a text box in future.
                print (f"Saved {saved_count}, Skipped {skipped_count}")
            # Return this anyway.
            return (f"Saved {saved_count}, Skipped {skipped_count}",)

        # We want to throw an error if something goes wrong.
        except Exception as e:
            print (f"Error saving captions: {e}")
            return (f"Error saving captions: {e}",)