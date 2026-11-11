from ..nodes.ui_feedback import feedback


class DebugModelNodeTpl:
    DESCRIPTION = "Pass through the pipeline and show model, task and device only. Full structured loading details are available from Model Loader's loading_diagnostics output."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_pipeline": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("debug_model", "debug_text")
    FUNCTION = "execute"
    CATEGORY = "Transformers Pipeline/Debug"

    def debug_pipeline(self, model_pipeline):
        model = getattr(model_pipeline, 'model', None)
        text = '\n'.join(f'{label}: {str(value)[:500]}' for label, value in (
            ('Model', getattr(model, 'name_or_path', 'unknown')),
            ('Task', getattr(model_pipeline, 'task', 'unknown')),
            ('Device', getattr(model_pipeline, 'device', 'unknown'))))
        return (model_pipeline, text)

    def execute(self, model_pipeline):
        result = self.debug_pipeline(model_pipeline)
        return feedback(result, result[1])
