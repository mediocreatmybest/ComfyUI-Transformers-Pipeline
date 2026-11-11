# A preset list of image-to-text models for the transformers pipeline

class PresetModelListTpl:
    DESCRIPTION = "Convenient model IDs, not automatic task selection or a compatibility guarantee. Connect selected_model to Model Loader; arbitrary IDs can also be typed there."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (cls.get_model_list(), {
                    "default": "Salesforce/blip-image-captioning-base",
                    "tooltip": "Model candidates, not a compatibility guarantee. BLIP and Florence community are tested on Desktop Transformers 4.57.3. Qwen3.5 needs the isolated 5.22.2 runtime; choosing it does not switch runtimes."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_model",)
    FUNCTION = "select_model"
    CATEGORY = "Transformers Pipeline"

    # Need to find additional models that work with pipeline like this.
    @staticmethod
    def get_model_list():
        """Return model candidates; architecture support depends on the runtime."""
        return [
            ("Salesforce/blip-image-captioning-base"),
            ("Salesforce/blip-image-captioning-large"),
            ("florence-community/Florence-2-base"),
            ("microsoft/Florence-2-base"),
            ("Qwen/Qwen3.5-4B"),
            ("Salesforce/blip2-opt-2.7b"),
            ("ybelkada/blip2-opt-2.7b-fp16-sharded"),
            ("Salesforce/blip2-opt-2.7b-coco"),
            ("Salesforce/blip2-opt-6.7b"),
            ("Salesforce/blip2-opt-6.7b-coco"),
            ("Salesforce/blip2-flan-t5-xl"),
            ("Salesforce/blip2-flan-t5-xxl"),
            ("Salesforce/blip2-flan-t5-xl-coco"),
            ("microsoft/git-base"),
            ("microsoft/git-large-coco"),
            ("nlpconnect/vit-gpt2-image-captioning"),
            ("Salesforce/instructblip-vicuna-7b"),
            ("Salesforce/instructblip-vicuna-13b"),
        ]

    def select_model(self, model_name):
        """ Returns the model as a simple string. """
        return (model_name,)

