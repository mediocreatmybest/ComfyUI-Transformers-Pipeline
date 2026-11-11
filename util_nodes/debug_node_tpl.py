import reprlib
from ..nodes.ui_feedback import feedback


class DebugNodeTpl:
    DESCRIPTION = "Pass through the original list and show a bounded summary. Does not print the full dataset to the console. Use Caption / Image Preview for caption records."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_list": ("LIST",)}}

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("debug_list", "debug_text")
    FUNCTION = "execute"
    CATEGORY = "Transformers Pipeline/Debug"

    def debug_data(self, input_list):
        formatter = reprlib.Repr()
        formatter.maxlist, formatter.maxdict = 5, 4
        formatter.maxstring = formatter.maxother = 300
        return (input_list, f'Items {len(input_list)}; bounded summary:\n{formatter.repr(input_list)[:4000]}')

    def execute(self, input_list):
        result = self.debug_data(input_list)
        return feedback(result, result[1])
