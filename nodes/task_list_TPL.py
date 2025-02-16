# A preset list of tasks that have been tested and seem to work with the transformers pipeline in these nodes.

class TaskListTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "task_list": (cls.get_task_list(), {
                    "default": "image-to-text",
                    "tooltip": "List of available tasks."
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
        """ Returns a list of supported tasks. """
        return [
            "image-to-text",
            "image-to-text-to-text",
        ]

    def select_task(self, task_list):  # fix for task_name to task_list
        """ Returns the selected task as a simple string. """
        return (task_list,)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "TaskListTpl": TaskListTpl
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "TaskListTpl": "Task List 📋"
}
