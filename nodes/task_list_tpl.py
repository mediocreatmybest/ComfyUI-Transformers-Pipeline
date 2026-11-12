# A preset list of tasks that have been tested and seem to work with the transformers pipeline in these nodes.

class TaskListTpl:
    DESCRIPTION = "Select a Transformers task string for Model Loader. Text-only tasks remain available for general pipelines, but Caption Generator requires an image-captioning or visual-QA task."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_list": (cls.get_task_list(), {
                    "default": "image-to-text",
                    "tooltip": "Tasks depend on your Transformers version. Caption Generator accepts image captioning and visual QA tasks."
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_task",)
    FUNCTION = "select_task"
    CATEGORY = "Transformers Pipeline"

    # Need to look at additional tasks that we can break down.
    @staticmethod
    def get_task_list():
        """Return task candidates; not every model or caption node supports every task."""
        return [
            "zero-shot-image-classification",
            "zero-shot-classification",
            "text-generation",
            "image-to-text",
            "image-text-to-text",
            "any-to-any",
            "visual-question-answering",
            "question-answering",
            "summarization",
            "translation",
        ]

    def select_task(self, task_list):  # fix for task_name to task_list
        """ Returns the selected task as a simple string. """
        return (task_list,)


