# Caption Generator generating the captions using pipeline
from .. import be_quiet  # Import config from root __init__.py

class CaptionGeneratorTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_pipeline": ("MODEL",),  # the main transformer pipeline from model loader
                "image_batches": ("LIST",),  # List of image in batches
            },
            "optional": {
                #"question": ("STRING", {"default": ""}),  # Optional question input for some models. Need to test...
                "prepend_text": ("STRING", {"default": ""}),  # Text before caption...
                "append_text": ("STRING", {"default": ""}),  # Text after caption...
                "max_new_tokens": ("INT", {"default": 75, "min": 15}),  # Token limit from pipeline to prevent warnings and limit output
            }
        }

    RETURN_TYPES = ("LIST",)  # Output: List of complete generated captions
    RETURN_NAMES = ("captions",)
    FUNCTION = "generate_captions"
    CATEGORY = "Transformers Pipeline"

    def generate_captions(self, model_pipeline, image_batches, question="", prepend_text="", append_text="", max_new_tokens=75):
        """
        Uses the loaded HF pipeline to generate captions.
        added questions and text append and prepend.
        """
        try:
            captions = []
            for batch in image_batches:
                if not be_quiet:
                    print(f"Processing Image Batch: {batch} | Question: {question}")

                # Format input for models that support text prompts usually blank
                input_data = [{"image": img, "text": question} for img in batch] if question else batch

                # Generate caption
                output = model_pipeline(input_data, max_new_tokens=max_new_tokens)

                if not be_quiet:
                    print(f"Raw Model Output: {output}")  # Debugging

                # Process different outputs depending on the model.
                processed_output = []
                if isinstance(output, str):
                    processed_output.append(output.strip())
                elif isinstance(output, list):
                    for sublist in output:
                        if isinstance(sublist, dict) and "generated_text" in sublist:
                            processed_output.append(sublist["generated_text"])
                        elif isinstance(sublist, list):
                            for item in sublist:
                                if isinstance(item, dict) and "generated_text" in item:
                                    processed_output.append(item["generated_text"])
                                elif isinstance(item, str):
                                    processed_output.append(item.strip())
                        elif isinstance(sublist, str):
                            processed_output.append(sublist.strip())

                # Apply prepend & append Text
                final_captions = [f"{prepend_text}{cap}{append_text}".strip() for cap in processed_output]
                captions.extend(final_captions)

            if not be_quiet:
                print(f"Extracted Captions: {captions}")  # Final check

            return (captions,)

        except Exception as e:
            print(f"Error generating captions: {e}")
            return ([],)

