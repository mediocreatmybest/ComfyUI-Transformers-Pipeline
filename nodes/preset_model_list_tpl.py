from ..captioning.catalogue import CATALOGUE, model_choices, preset_model


class PresetModelListTpl:
    DESCRIPTION = "Pick a model ID to connect to Model Loader in Custom mode. For a matching model and task in one control, use Model Preset on Model Loader."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model_name": (model_choices()[1:] + cls.get_model_list(), {
                "default": "Salesforce/blip-image-captioning-base",
                "tooltip": "Grouped catalogue choices appear first. Original IDs remain available for saved workflows.",
            }),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_model",)
    FUNCTION = "select_model"
    CATEGORY = "Transformers Pipeline"

    @staticmethod
    def get_model_list():
        return [model["id"] for model in CATALOGUE.values() if model.get("legacy")]

    def select_model(self, model_name):
        # Legacy API callers can still pass through a literal ID or path.
        model = preset_model(model_name) if model_name in model_choices()[1:] else None
        return (model["id"] if model else model_name,)
