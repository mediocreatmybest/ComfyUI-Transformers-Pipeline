from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from .records import CaptionRecord, publish_file, write_records


def _txt(items, staging):
    for record, candidate, name in items:
        if name:
            (staging / Path(name).with_suffix(".txt")).write_text(candidate.text + "\n", encoding="utf-8")


def _imagefolder(items, staging):
    rows = [{"file_name": name, "text": candidate.text, "tags": candidate.tags,
             "image_id": record.image.id, "candidate_id": candidate.id, "profile_id": candidate.profile_id}
            for record, candidate, name in items]
    (staging / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _native(items, staging):
    write_records([replace(record, candidates=[candidate]) for record, candidate, _ in items], staging / "records.jsonl")


# Adapters consume the same selected records and filenames; new formats need not change records.
ADAPTERS = {"txt": _txt, "imagefolder": _imagefolder, "native": _native}


def export_records(records, destination="caption_exports", format="txt", *, profile_id,
                   caption_type=None, candidate_ids=None, overwrite=False, approved_only=False):
    if format not in ADAPTERS or not profile_id:
        raise ValueError("Select a known export format and an explicit profile identifier")
    destination = Path(destination).resolve()
    candidate_ids = candidate_ids or {}
    records = [CaptionRecord.from_dict(r.to_dict()) for r in records]
    ids = [r.image.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Merge duplicate image records before export")
    if set(candidate_ids) - set(ids):
        raise ValueError("Candidate selection contains unknown image identifiers")
    items, skipped, names = [], [], set()
    for record in records:
        source = Path(record.image.source_path).resolve()
        if source.is_relative_to(destination):
            raise ValueError("Export destination must not contain the original image collection")
        candidates = [c for c in record.candidates if c.profile_id == profile_id and
                      (caption_type is None or c.caption_type == caption_type)]
        if record.image.id in candidate_ids:
            candidates = [c for c in candidates if c.id == candidate_ids[record.image.id]]
            if not candidates:
                raise ValueError("Selected candidate does not match its image/profile/type")
        if len(candidates) > 1:
            raise ValueError(f"Ambiguous candidates for {source}; select a candidate id")
        if not candidates:
            skipped.append({"image_id": record.image.id, "reason": "no matching profile/type"})
            continue
        candidate = candidates[0]
        if (approved_only and candidate.approval != "approved") or (format != "native" and (
            candidate.status != "success" or candidate.approval == "rejected"
        )):
            skipped.append({"image_id": record.image.id, "reason": candidate.error or candidate.approval})
            continue
        name = None
        if candidate.status == "success":
            if hashlib.sha256(source.read_bytes()).hexdigest() != record.image.sha256:
                raise ValueError(f"Source image changed since captioning: {source}")
            suffix = source.suffix.lower()
            if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}:
                raise ValueError(f"Unsupported image suffix: {suffix}")
            stem = re.sub(r"[^\w-]", "_", source.stem)[:60]
            name = f"{stem}-{record.image.id[:20]}{suffix}"
            if name.casefold() in names:
                raise ValueError("Export filename collision")
            names.add(name.casefold())
        items.append((record, candidate, name))
    if not items:
        raise ValueError(f"No eligible candidates for profile {profile_id!r}; nothing exported")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="caption-export-", dir=destination.parent) as folder:
        staging = Path(folder)
        for record, candidate, name in items:
            if name:
                shutil.copyfile(record.image.source_path, staging / name)
                if hashlib.sha256((staging / name).read_bytes()).hexdigest() != record.image.sha256:
                    raise ValueError("Source changed during export; nothing written to destination")
        ADAPTERS[format](items, staging)
        files = sorted(staging.iterdir())
        if destination.exists():
            existing = {file.name for file in destination.iterdir()}
            if existing and existing != {file.name for file in files}:
                raise ValueError("Dataset membership changed; use a new destination to avoid stale training pairs")
        for file in files:
            target = destination / file.name
            if target.is_symlink() or target.is_dir():
                raise ValueError(f"Unsafe export target: {target}")
            if target.exists() and not overwrite:
                raise FileExistsError(f"Export exists; use a new destination or explicit overwrite: {target}")
        destination.mkdir(parents=True, exist_ok=True)
        created = []
        try:
            for file in files:
                target = destination / file.name
                publish_file(file, target, overwrite)
                if not overwrite:
                    created.append(target)
        except Exception:
            for target in created:
                target.unlink()
            raise
    return {"format": format, "destination": str(destination), "exported": len(items),
            "skipped": skipped, "files": [file.name for file in files]}
