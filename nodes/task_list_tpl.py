class TaskListTpl:
    DESCRIPTION = "Advanced task picker for Model Loader in Custom mode. Model Preset chooses a task automatically. The loader checks the installed Transformers runtime."
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_list": (cls.get_task_list(), {
                    "default": "image-to-text",
                    "tooltip": "The available tasks depend on the installed Transformers version. Text tasks do not accept images."
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


