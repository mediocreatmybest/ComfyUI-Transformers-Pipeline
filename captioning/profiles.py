from dataclasses import dataclass, field
import json
from pathlib import Path


SAFETY_PROMPT = (
    "Describe only visible evidence. Do not identify or infer the creator or artist. "
    "Do not invent unseen objects, history, intent, or precise materials. "
    "Qualify uncertain visual interpretations; describe medium as apparent, not verified."
)


@dataclass
class CaptionProfile:
    id: str
    caption_type: str
    prompt: str
    max_new_tokens: int = 192
    pipeline_kwargs: dict = field(default_factory=dict)
    output_format: str = "text"
    prompt_mode: str = "instruction"

    def __post_init__(self):
        if not self.id or not self.caption_type or not isinstance(self.prompt, str):
            raise ValueError("Profiles require id, caption_type and a text prompt")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.output_format not in {"text", "tags"} or self.prompt_mode not in {"instruction", "raw"}:
            raise ValueError("Invalid profile output_format or prompt_mode")
        validate_options(self.pipeline_kwargs)

    @property
    def effective_prompt(self):
        return f"{self.prompt}\n{SAFETY_PROMPT}" if self.prompt_mode == "instruction" else self.prompt


def validate_options(options):
    if not isinstance(options, dict):
        raise ValueError("Pipeline options must be a JSON object")
    reserved = {"images", "image", "text", "question", "prompt", "max_new_tokens", "return_full_text",
                "return_tensors", "return_type", "skip_special_tokens"}
    if reserved.intersection(options):
        raise ValueError("Pipeline options cannot replace image, prompt, token limit or output format")
    json.dumps(options, allow_nan=False)


def load_profiles(path=None, *, data=None):
    if data is None:
        path = Path(path) if path else Path(__file__).with_name("profiles.json")
        data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported profile schema version")
    profiles = {}
    for value in data["profiles"]:
        profile = CaptionProfile(**value)
        if profile.id in profiles:
            raise ValueError(f"Duplicate profile id: {profile.id}")
        profiles[profile.id] = profile
    if not profiles:
        raise ValueError("At least one profile is required")
    return profiles


def select_profiles(identifiers, path=None, data=None):
    available = load_profiles(path, data=data)
    ids = [part.strip() for part in identifiers.split(",") if part.strip()]
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(available):
        raise ValueError(f"Select distinct profile ids from: {', '.join(available)}")
    return [available[name] for name in ids]
