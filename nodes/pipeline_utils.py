from transformers import __version__ as transformers_version

from ..captioning.pipeline import CAPTION_TASKS, caption_text
from ..captioning.pipeline import caption_image as _caption_image


def caption_image(pipe, source, question, max_new_tokens, pipeline_kwargs=None):
    return _caption_image(pipe, source, question, max_new_tokens, pipeline_kwargs,
                          runtime_version=transformers_version)
