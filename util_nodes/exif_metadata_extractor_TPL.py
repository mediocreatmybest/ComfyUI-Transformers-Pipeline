from PIL import Image
import exifread
import os

class ExifMetadataExtractorTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("LIST",),  # List of image file paths
            }
        }

    INPUT_LABELS = {
        "image_paths": "Image Paths",
    }

    RETURN_TYPES = ("LIST", "LIST")  # Returns extracted metadata as a list
    RETURN_NAMES = ("metadata", "formatted_metadata")
    FUNCTION = "extract_exif_metadata"
    CATEGORY = "Transformers Pipeline"

    def extract_exif_metadata(self, image_paths):
        """
        Extracts EXIF metadata from a list of image files using the image loader node.
        Skips tags with no data and writing empty files.
        """
        metadata_list = []
        formatted_list = []

        try:
            for image_path in image_paths:
                metadata = {}

                # Open the image and extract EXIF data with exifread
                with open(image_path, "rb") as img_file:
                    tags = exifread.process_file(img_file)

                # Extract EXIF fields and add if data is available, what other tags we should collect? Make a bool option?
                if "EXIF ISOSpeedRatings" in tags:
                    metadata["ISO"] = tags["EXIF ISOSpeedRatings"]
                if "EXIF FNumber" in tags:
                    metadata["Aperture"] = tags["EXIF FNumber"]
                if "EXIF ExposureTime" in tags:
                    metadata["Shutter Speed"] = tags["EXIF ExposureTime"]
                if "EXIF FocalLength" in tags:
                    metadata["Focal Length"] = tags["EXIF FocalLength"]
                if "Image Model" in tags:
                    metadata["Camera Model"] = tags["Image Model"]
                if "EXIF LensModel" in tags:
                    metadata["Lens Model"] = tags["EXIF LensModel"]
                #if "EXIF DateTimeOriginal" in tags:
                #    metadata["Date Taken"] = tags["EXIF DateTimeOriginal"]

                # If no metadata is found, skip the image
                if not metadata:
                    continue

                metadata_list.append(metadata)

                # Create a formatted string output
                formatted = ", ".join(f"{key}: {value}" for key, value in metadata.items())
                formatted_list.append(formatted)

            return (metadata_list, formatted_list)

        except Exception as e:
            return ([{"Error": str(e)}], [f"Error extracting metadata: {e}"])
