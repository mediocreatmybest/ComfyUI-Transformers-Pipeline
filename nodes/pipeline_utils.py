"""Keep image/prompt and text conventions at the Transformers boundary."""

from copy import deepcopy

from PIL import Image, ImageOps
from transformers import __version__ as transformers_version


CAPTION_TASKS = {"image-to-text", "image-text-to-text", "any-to-any", "visual-question-answering"}


def caption_text(output):
    if isinstance(output, list):
        if output and all(isinstance(item, dict) and "role" in item for item in output):
            assistants = [item["content"] for item in output if item["role"] == "assistant"]
            if not assistants:
                raise ValueError("Output contains no assistant response")
            content = assistants[-1]
            if isinstance(content, list):
                content = "\n".join(item["text"] for item in content if item.get("type") == "text")
            return caption_text(content)
        if len(output) != 1:
            raise ValueError(f"Expected one caption per image, received {len(output)} results")
        return caption_text(output[0])
    if isinstance(output, dict):
        for key in ("generated_text", "answer"):
            if key in output:
                return caption_text(output[key])
    if isinstance(output, str) and output.strip():
        return output.strip()
    raise ValueError("Pipeline returned an empty or unsupported caption output")


def caption_image(pipe, source, question, max_new_tokens, pipeline_kwargs=None):
    # Some pipelines mutate generation/processor kwargs during preprocessing.
    extra = deepcopy(pipeline_kwargs or {})
    if isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source).convert("RGB")
    else:
        with Image.open(source) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
    try:
        task = pipe.task
        if task == "image-to-text":
            options = {**extra, "max_new_tokens": max_new_tokens}
            if question:
                options["prompt"] = question
            output = pipe(image, **options)
        elif task == "visual-question-answering":
            if not question.strip():
                raise ValueError("Visual question answering requires a question")
            extra["top_k"] = 1
            output = pipe(image=image, question=question, **extra)
        else:
            processor = pipe.processor
            model_type = getattr(pipe.model.config, "model_type", "")
            florence = model_type == "florence2"
            prompt = question or ("<CAPTION>" if florence else "Describe this image.")
            if model_type in {"blip", "blip-2", "git", "vision-encoder-decoder"}:
                prompt = question
            options = {**extra, "max_new_tokens": max_new_tokens, "return_full_text": False}
            if florence:
                output = pipe(images=image, text=prompt, skip_special_tokens=False, **options)
                raw = caption_text(output)
                parsed = processor.post_process_generation(raw, task=prompt, image_size=image.size)
                if prompt not in parsed or not isinstance(parsed[prompt], str):
                    raise ValueError(f"Florence task {prompt} did not return a text caption")
                return caption_text(parsed[prompt])
            if getattr(processor, "chat_template", None):
                if int(transformers_version.split(".")[0]) < 5:
                    supported = {"generate_kwargs", "stop_sequence", "continue_final_message", "timeout", "clean_up_tokenization_spaces"}
                    if extra.keys() - supported:
                        raise ValueError("Chat template/processor options require Transformers 5; Transformers 4 ignores them")
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]}]
                output = pipe(text=messages, **options)
            else:
                output = pipe(images=image, text=prompt, **options)
        return caption_text(output)
    finally:
        image.close()
