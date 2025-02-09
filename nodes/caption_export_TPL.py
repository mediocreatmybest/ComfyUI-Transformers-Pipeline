# Caption Exporter - Save captions to text file

import os

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
                "keep_extension": ("BOOLEAN", {"default": True}),  # Toggle for filename format
                "overwrite_existing": ("BOOLEAN", {"default": False}),  # Overwrite or skip
                "allow_empty_files": ("BOOLEAN", {"default": False}),  # Do we want to allow empty files or skip
            }
        }

    INPUT_LABELS = {
        "captions": "Generated Captions",
        "image_paths": "Image Paths",
        "keep_extension": "Keep File Extension",
        "overwrite_existing": "Overwrite Existing Captions",
        "allow_empty_files": "Allow Saving Empty Files",
    }

    RETURN_TYPES = ("STRING",)  # Outputs a message confirming save status - success or not - Make this optional?
    RETURN_NAMES = ("status",)
    FUNCTION = "save_captions"
    CATEGORY = "Transformers Pipeline"

    def save_captions(self, captions, image_paths, keep_extension=True, overwrite_existing=False, allow_empty_files=False):
        """
        Saves captions as text files next to their paired image.
        Skips empty captions set as default unless 'allow_empty_files' is true.
        """
        try:
            if not captions:
                return ("No captions, skipping.",)

            saved_count = 0
            skipped_count = 0

            for caption, image_path in zip(captions, image_paths):
                # Skip saving if caption is empty and allow_empty_files is False, we may want blank files for some reason
                if not caption.strip() and not allow_empty_files:
                    skipped_count += 1
                    continue

                # Extract filename
                base_name = os.path.basename(image_path)
                file_name = f"{base_name}.txt" if keep_extension else f"{os.path.splitext(base_name)[0]}.txt"
                caption_path = os.path.join(os.path.dirname(image_path), file_name)

                # Skip if file exists and overwrite is disabled
                if os.path.exists(caption_path) and not overwrite_existing:
                    skipped_count += 1
                    continue

                # Save caption file at utf-8 to allow special characters
                with open(caption_path, "w", encoding="utf-8") as f:
                    f.write(caption.strip() + "\n")
                    saved_count += 1

            return (f"Saved {saved_count}, Skipped {skipped_count}",)

        except Exception as e:
            return (f"Error saving captions: {e}",)
