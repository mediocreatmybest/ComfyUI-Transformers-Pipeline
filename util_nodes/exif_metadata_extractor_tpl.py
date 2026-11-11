from ..nodes.ui_feedback import feedback


class ExifMetadataExtractorTpl:
    DESCRIPTION = "Extract camera EXIF without decoding pixels. Each input keeps one output position: empty metadata is {}, and read failures contain Error. Requires optional exifread."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_paths": ("LIST",)}}

    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("metadata", "formatted_metadata")
    FUNCTION = "execute"
    CATEGORY = "Transformers Pipeline"

    def extract_exif_metadata(self, image_paths):
        try:
            import exifread
        except ImportError as exc:
            raise RuntimeError("EXIF extraction requires the optional exifread package.") from exc
        fields = {"EXIF ISOSpeedRatings": "ISO", "EXIF FNumber": "Aperture",
                  "EXIF ExposureTime": "Shutter Speed", "EXIF FocalLength": "Focal Length",
                  "Image Model": "Camera Model", "EXIF LensModel": "Lens Model"}
        metadata_list, formatted_list = [], []
        for path in image_paths:
            try:
                with open(path, 'rb') as stream:
                    tags = exifread.process_file(stream)
                metadata = {label: tags[tag] for tag, label in fields.items() if tag in tags}
                formatted = ', '.join(f'{key}: {value}' for key, value in metadata.items())
            except Exception as exc:
                metadata = {'Error': f'{path}: {exc}'}
                formatted = f'Error extracting metadata: {path}: {exc}'
            metadata_list.append(metadata)
            formatted_list.append(formatted)
        return (metadata_list, formatted_list)

    def execute(self, image_paths):
        result = self.extract_exif_metadata(image_paths)
        failed = sum('Error' in item for item in result[0])
        empty = sum(not item for item in result[0])
        return feedback(result, f'Images {len(image_paths)}, With metadata {len(image_paths)-empty-failed}, Empty {empty}, Failed {failed}. Output positions match input positions.')
