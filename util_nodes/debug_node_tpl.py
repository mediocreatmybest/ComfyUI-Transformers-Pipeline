# Lets sit between nodes to see what is happening
class DebugNodeTpl:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("LIST",),  # Accepts any list, add optional strings in future
            }
        }

    INPUT_LABELS = {
        "input_list": "Data to Debug",
    }

    RETURN_TYPES = ("LIST", "STRING")  # Returns both raw and readable data
    RETURN_NAMES = ("debug_list", "debug_text")
    FUNCTION = "debug_data"
    CATEGORY = "Transformers Pipeline/Debug"

    def debug_data(self, input_list):
        """
        Passes the input list through unchanged and then displays its content as a simple string.
        """
        # We should be able to remove the return list string, as nodes can connect to multiple nodes.
        # Something else for the todo list
        debug_text = f"Debug Output: {input_list}"
        print(debug_text)  # Prints to console for logging - We are using a debug node after all
        return (input_list, debug_text)
