from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import uuid

from PIL import Image, ImageOps

from .profiles import validate_options
from .records import CaptionCandidate, CaptionRecord, ImageIdentity, ModelInfo


def runtime_versions():
    versions = {"python": platform.python_version()}
    for name in ("transformers", "torch", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def generate_records(pipe, image_paths, profiles, *, model_id=None, revision=None,
                     pipeline_kwargs=None, authors=None, infer=None):
    if infer is None:
        from .pipeline import caption_image
        infer = caption_image
    profiles = list(profiles)
    if not profiles or len({p.id for p in profiles}) != len(profiles):
        raise ValueError("Select one or more distinct profile identifiers")
    pipeline_kwargs = {} if pipeline_kwargs is None else pipeline_kwargs
    validate_options(pipeline_kwargs)
    authors = {str(Path(path).resolve()): author for path, author in (authors or {}).items()}
    config = getattr(pipe.model, "config", None)
    provenance = getattr(pipe, "_tpl_model_info", {})
    model = ModelInfo(model_id or provenance.get("model_id") or pipe.model.name_or_path,
                      revision or provenance.get("revision") or getattr(config, "_commit_hash", None),
                      pipe.task, {**runtime_versions(), **({"adapter": deepcopy(provenance["adapter"])} if provenance.get("adapter", {}).get("active") else {})})
    records = []
    for source in image_paths:
        path = Path(source).resolve()
        digest = None
        image = None
        error = None
        author = authors.get(str(path))
        author_source = "supplied" if author else None
        try:
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            with Image.open(io.BytesIO(content)) as original:
                if not author:
                    author = original.getexif().get(315) or original.info.get("Artist") or original.info.get("Author")
                    if author:
                        author = str(author).strip()
                        author_source = "source_metadata"
                image = ImageOps.exif_transpose(original).convert("RGB")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        identity = hashlib.sha256((os.path.normcase(str(path)) + "\0" + (digest or "")).encode("utf-8")).hexdigest()
        record = CaptionRecord(ImageIdentity(identity, str(path), digest,
                                             image.width if image else None, image.height if image else None,
                                             author, author_source), [])
        try:
            for profile in profiles:
                options = {**deepcopy(profile.pipeline_kwargs), **deepcopy(pipeline_kwargs)}
                candidate = dict(id=uuid.uuid4().hex, caption_type=profile.caption_type, profile_id=profile.id,
                                 prompt=profile.effective_prompt, generation_parameters={"max_new_tokens": profile.max_new_tokens,
                                 "pipeline_kwargs": options, "prompt_mode": profile.prompt_mode}, model=deepcopy(model))
                generated = ""
                try:
                    if error:
                        raise ValueError(error)
                    if profile.prompt_mode == "instruction" and (
                        pipe.task == "image-to-text" or getattr(config, "model_type", "") in {"blip", "blip-2", "git", "vision-encoder-decoder", "florence2"}
                    ):
                        raise ValueError("This backend requires a raw caption-prefix/task-token profile, or use an instruction-following vision-language model")
                    generated = infer(pipe, image, profile.effective_prompt, profile.max_new_tokens, deepcopy(options))
                    if not isinstance(generated, str) or not generated.strip():
                        raise ValueError("Pipeline returned no caption text")
                    tags = []
                    text = generated.strip()
                    if profile.output_format == "tags":
                        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
                        tags = json.loads(fenced.group(1) if fenced else text)
                        if not isinstance(tags, list) or not tags or any(not isinstance(t, str) or not t.strip() for t in tags):
                            raise ValueError("Tag profile requires a nonempty JSON array of strings")
                        tags = list(dict.fromkeys(t.strip() for t in tags))
                        text = ", ".join(tags)
                    result = CaptionCandidate(**candidate, status="success", text=text, tags=tags, generated_text=generated)
                except Exception as exc:
                    result = CaptionCandidate(**candidate, status="failure", error=f"{type(exc).__name__}: {exc}", generated_text=generated)
                record.candidates.append(result)
        finally:
            if image:
                image.close()
        records.append(record)
    return records
