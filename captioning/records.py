from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


@dataclass
class ImageIdentity:
    id: str
    source_path: str
    sha256: str | None
    width: int | None = None
    height: int | None = None
    author: str | None = None
    author_source: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not re.fullmatch(r"[0-9a-f]{64}", self.id):
            raise ValueError("Image id must be a SHA-256 identity")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("Invalid image file hash")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise ValueError("Source path is required")
        if (self.width is None) != (self.height is None):
            raise ValueError("Both image dimensions are required together")
        if self.width is not None and (type(self.width) is not int or type(self.height) is not int or min(self.width, self.height) < 1):
            raise ValueError("Invalid image dimensions")


@dataclass
class ModelInfo:
    model_id: str
    revision: str | None
    pipeline_task: str
    runtime: dict = field(default_factory=dict)


@dataclass
class CaptionCandidate:
    id: str
    caption_type: str
    profile_id: str
    prompt: str
    generation_parameters: dict
    model: ModelInfo
    status: str
    text: str = ""
    tags: list[str] = field(default_factory=list)
    error: str | None = None
    approval: str = "unreviewed"
    edited: bool = False
    generated_text: str = ""

    def __post_init__(self):
        if self.status not in {"success", "failure"} or self.approval not in {"unreviewed", "approved", "rejected"}:
            raise ValueError("Invalid candidate status or approval")
        if self.status == "success" and (not isinstance(self.text, str) or not self.text.strip() or self.error):
            raise ValueError("Successful candidates require nonempty text and no error")
        if self.status == "failure" and (self.text or self.tags or not self.error or self.approval != "unreviewed"):
            raise ValueError("Failed candidates must have an error, no caption/tags, and no approval")
        if not isinstance(self.tags, list) or any(not isinstance(tag, str) or not tag.strip() for tag in self.tags):
            raise ValueError("Tags must be a list of nonempty strings")
        if not self.id or not self.profile_id or not self.caption_type:
            raise ValueError("Candidate identity, profile and type are required")

    def edit(self, text, tags=None):
        if self.status != "success" or not isinstance(text, str) or not text.strip():
            raise ValueError("Only successful candidates can be edited to nonempty text")
        if tags is not None and (not isinstance(tags, list) or any(not isinstance(t, str) or not t.strip() for t in tags)):
            raise ValueError("Tags must be nonempty strings")
        self.text = text.strip()
        self.tags = list(tags) if tags is not None else []
        self.edited = True
        self.approval = "unreviewed"

    def review(self, approval):
        if self.status != "success" or approval not in {"unreviewed", "approved", "rejected"}:
            raise ValueError("Review requires a successful candidate and a valid approval status")
        self.approval = approval


@dataclass
class CaptionRecord:
    image: ImageIdentity
    candidates: list[CaptionCandidate]
    schema_version: int = 1

    def __post_init__(self):
        if self.schema_version != 1:
            raise ValueError("Unsupported caption record schema version")
        if any(c.status == "success" for c in self.candidates) and (not self.image.sha256 or not self.image.width):
            raise ValueError("Successful candidates require an image hash and dimensions")
        if len({c.id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate candidate ids")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("schema_version", None) != 1:
            raise ValueError("Unsupported caption record schema version")
        image = ImageIdentity(**value.pop("image"))
        candidates = []
        for item in value.pop("candidates"):
            item = dict(item)
            item["model"] = ModelInfo(**item["model"])
            candidates.append(CaptionCandidate(**item))
        if value:
            raise ValueError(f"Unknown record fields: {sorted(value)}")
        return cls(image, candidates)


def iter_records(path):
    """Stream validated records so preview need not retain a whole dataset."""
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield CaptionRecord.from_dict(json.loads(line))
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(f"Invalid caption record at line {number}: {exc}") from exc


def read_records(path):
    return list(iter_records(path))


def write_records(records, path, overwrite=False):
    path = Path(path)
    records = list(records)
    if any(path.resolve() == Path(r.image.source_path).resolve() for r in records):
        raise ValueError("A record file cannot replace a source image")
    # Validate before creating or replacing a file.
    lines = [json.dumps(CaptionRecord.from_dict(r.to_dict()).to_dict(), ensure_ascii=False, allow_nan=False) for r in records]
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write("".join(line + "\n" for line in lines))
    try:
        publish_file(temporary, path, overwrite)
    finally:
        temporary.unlink(missing_ok=True)


def publish_file(source, target, overwrite=False):
    if overwrite:
        os.replace(source, target)
        return
    # Exclusive creation works on filesystems without hardlink support.
    with open(target, "xb") as output:
        try:
            with open(source, "rb") as stream:
                shutil.copyfileobj(stream, output)
        except Exception:
            output.close()
            Path(target).unlink()
            raise
