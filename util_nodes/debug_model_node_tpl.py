# Debug the Pipeline
import pprint
from transformers import Pipeline

class DebugModelNodeTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_pipeline": ("MODEL",),  # Expecting a loaded pipeline
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")  # Returns the model and string
    RETURN_NAMES = ("debug_model", "debug_text")
    FUNCTION = "debug_pipeline"
    CATEGORY = "Transformers Pipeline/Debug"

    def debug_pipeline(self, model_pipeline):
        """
        debug the pipeline and display all details including config.
        """
        try:
            if not isinstance(model_pipeline, Pipeline):
                debug_text = "Warning: Input is not a Transformers pipeline..."
            else:
                debug_text = (
                    f"\n|====================>\n"
                    f"Debugging Pipeline:\n"
                    f"Model: {model_pipeline.model.name_or_path}\n"
                    f"Task Type: {model_pipeline.task}\n"
                    f"Device: {model_pipeline.device}\n"
                    f"Config: {model_pipeline.model.config.to_dict()}"
                    f"\n<====================|\n"
                )

            # Print to console
            print(debug_text)
            # auto import and use pretty print
            #pprint.pprint(debug_text)

            return (model_pipeline, debug_text)

        except Exception as e:
            error_msg = f"Error in Debug Pipeline Node: {e}"
            print(error_msg)  # Throw error to the console
            return (model_pipeline, error_msg)

