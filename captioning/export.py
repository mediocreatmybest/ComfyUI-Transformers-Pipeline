from contextlib import ExitStack
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from threading import Lock
from uuid import uuid4

from .records import CaptionRecord, publish_file, read_records, write_records
from .provenance import caption_content, source_issue


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
PUBLISH_LOCK = Lock()


def _txt(items, staging):
    for _, candidate, name in items:
        (staging / Path(name).with_suffix(".txt")).write_text(caption_content(candidate) + "\n", encoding="utf-8")


def _imagefolder(items, staging):
    rows = [{"file_name": name, "text": caption_content(candidate), "tags": candidate.tags,
             "image_id": record.image.id, "candidate_id": candidate.id, "profile_id": candidate.profile_id}
            for record, candidate, name in items]
    (staging / "metadata.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _spreadsheet_cell(value):
    if value.startswith(("\t", "\r", "\n")) or value.lstrip().startswith(("=", "+", "-", "@", "＝", "＋", "－", "＠")):
        return "'" + value
    return value


def _csv(items, staging):
    with (staging / "captions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("source_filename", "caption", "exported_image", "profile", "review", "tags"))
        for record, candidate, name in items:
            writer.writerow(map(_spreadsheet_cell, (
                Path(record.image.source_path).name, caption_content(candidate), name or "",
                candidate.profile_id, candidate.approval,
                json.dumps(candidate.tags, ensure_ascii=False) if candidate.tags else "",
            )))


def _native(items, staging):
    # Legacy filtered v1 exports retain one candidate. Full Records retains all versions verbatim.
    write_records([record if candidate is None or record.schema_version == 2 else replace(record, candidates=[candidate])
                   for record, candidate, _ in items], staging / "records.jsonl")


ADAPTERS = {"txt": _txt, "imagefolder": _imagefolder, "csv": _csv, "native": _native}


def load_export_records(caption_records, records_file):
    if (caption_records is None) == (not records_file):
        raise ValueError("Supply connected caption records or a saved records file, exclusively")
    return read_records(records_file) if records_file else [
        CaptionRecord.from_dict(r.to_dict() if hasattr(r, 'to_dict') else r) for r in caption_records
    ]


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(path, *, directory=False):
    for part in (*path.parents, path):
        if part.is_symlink() or getattr(part, 'is_junction', lambda: False)():
            raise ValueError(f"Unsafe export target (symlink or junction): {part}")
    if path.exists() and not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Unsafe export target: {path}")


def _publish(files, overwrite, reserved):
    backups, completed = {}, []
    for staged, target in files:
        _safe_path(target)
        if target.resolve() in reserved:
            raise ValueError(f"Export target would replace an input file: {target}")
        if target.exists():
            if not overwrite:
                raise FileExistsError(f"Export already exists: {target}")
            backup = staged.with_name(staged.name + '.backup')
            shutil.copy2(target, backup)
            backups[target] = backup
    try:
        for staged, target in files:
            _safe_path(target)
            publish_file(staged, target, overwrite=target in backups)
            completed.append(target)
    except Exception:
        for target in reversed(completed):
            if target in backups:
                os.replace(backups[target], target)
            else:
                target.unlink(missing_ok=True)
        raise


def _separate_export(items, destination, format, copy_images, overwrite, version, reserved, friendly):
    if friendly and not version and not overwrite and destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output folder already contains files: {destination}")
    with ExitStack() as stack:
        created = []
        try:
            missing = []
            parent = destination.parent
            while not parent.exists():
                missing.append(parent)
                parent = parent.parent
            for folder in reversed(missing):
                folder.mkdir()
                created.append(folder)
            if version:
                destination = Path(tempfile.mkdtemp(prefix=destination.name + '-v-', dir=destination.parent))
                created.append(destination)
            staging = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='.caption-export-', dir=destination.parent)))
            if copy_images:
                for record, _, name in items:
                    if name:
                        shutil.copyfile(record.image.source_path, staging / name)
                        if _hash(staging / name) != record.image.sha256:
                            raise ValueError("Source changed during export; nothing written to destination")
            ADAPTERS[format](items, staging)
            files = sorted(staging.iterdir())
            _safe_path(destination, directory=True)
            if destination.exists():
                existing = {file.name for file in destination.iterdir()}
                if friendly and existing and not overwrite:
                    raise FileExistsError(f"Output folder already contains files: {destination}")
                if existing and existing != {file.name for file in files}:
                    raise ValueError("Dataset membership changed; choose a new destination to avoid stale training pairs")
            else:
                destination.mkdir()
                created.append(destination)
            _publish([(file, destination / file.name) for file in files], overwrite, reserved)
        except Exception:
            stack.close()
            for folder in reversed(created):
                # If another process added a file, leave its directory intact.
                if not any(folder.iterdir()):
                    folder.rmdir()
            raise
    return destination, [file.name for file in files]


def _sidecar_export(items, overwrite, version, reserved):
    version_suffix = '.v-' + uuid4().hex[:12] if version else ''
    with ExitStack() as stack:
        files = []
        seen = set()
        staging_by_parent = {}
        for record, candidate, _ in items:
            source = Path(record.image.source_path)
            target = source.with_name(source.name + version_suffix + '.txt')
            _safe_path(target)
            key = str(target.resolve()).casefold()
            if key in seen:
                raise ValueError(f"Caption destination collision: {target}")
            seen.add(key)
            if target.parent not in staging_by_parent:
                staging_by_parent[target.parent] = Path(stack.enter_context(
                    tempfile.TemporaryDirectory(prefix='.caption-export-', dir=target.parent)))
            staged = staging_by_parent[target.parent] / target.name
            staged.write_text(caption_content(candidate) + '\n', encoding='utf-8')
            files.append((staged, target))
        _publish(files, overwrite, reserved)
    return [str(target) for _, target in files]


def export_records(records, destination="caption_exports", format="txt", *, profile_id=None,
                   caption_type=None, candidate_ids=None, overwrite=False, approved_only=False,
                   copy_images=None, beside=False, version=False, archive_all=False, allow_empty=False,
                   input_file=None):
    """Select training text or archive record history and publish a prevalidated batch."""
    if format not in ADAPTERS:
        raise ValueError("Select a known export format")
    if copy_images is None:
        copy_images = not beside
    if beside and format != 'txt':
        raise ValueError("Choose a separate Output Folder for CSV, ImageFolder or Full Records. Beside originals supports TXT sidecars only.")
    if beside and copy_images:
        raise ValueError('Turn off Copy Images to save TXT sidecars beside the originals.')
    if format == 'imagefolder' and not copy_images:
        raise ValueError("ImageFolder requires Copy Images so each metadata row has a matching image.")
    if version and overwrite:
        raise ValueError("Choose either a new version or replacement of existing output")
    if not beside and not str(destination).strip():
        raise ValueError("Choose an Output Folder")
    destination = Path(os.path.abspath(Path(destination).expanduser()))
    if not beside:
        _safe_path(destination, directory=True)
    candidate_ids = {} if candidate_ids is None else candidate_ids
    if not isinstance(candidate_ids, dict) or any(not isinstance(value, str) for value in candidate_ids.values()):
        raise ValueError("Candidate IDs must map image IDs to caption candidate IDs")
    records = [CaptionRecord.from_dict(r.to_dict()) for r in records]
    ids = {r.image.id for r in records}
    if len(ids) != len(records):
        raise ValueError("Merge duplicate image records before export")
    if set(candidate_ids) - ids:
        raise ValueError("Candidate selection contains unknown image identifiers")
    sources = [Path(record.image.source_path).resolve() for record in records]
    if not beside and any(source.is_relative_to(destination) for source in sources):
        raise ValueError("Export destination must not contain the original image collection")
    reserved = set(sources)
    if input_file:
        reserved.add(Path(input_file).resolve())
    full_archive = format == 'native' and archive_all
    items, skipped, names = [], [], set()
    for record, source in zip(records, sources):
        candidate = None
        if not full_archive:
            candidates = [c for c in record.candidates if (not profile_id or c.profile_id == profile_id) and
                          (not caption_type or c.caption_type == caption_type)]
            if record.image.id in candidate_ids:
                candidates = [c for c in candidates if c.id == candidate_ids[record.image.id]]
                if not candidates:
                    raise ValueError("Selected candidate does not match its image/profile/type")
            if len(candidates) > 1:
                profiles = ', '.join(sorted({c.profile_id for c in candidates}))
                raise ValueError(f"Ambiguous captions for {source.name}. Select a Caption Style ({profiles}) or a candidate ID.")
            if candidates:
                candidate = candidates[0]
            reason = _skip_reason(candidate, record, format, approved_only)
            if reason:
                skipped.append({"image_id": record.image.id, "source_filename": source.name, "reason": reason})
                continue
        name = None
        if (full_archive and copy_images and record.image.sha256 and record.image.width) or (candidate and candidate.status == "success"):
            if _hash(source) != record.image.sha256:
                raise ValueError(f"Source image changed since captioning: {source}")
            suffix = source.suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                raise ValueError(f"Unsupported image suffix: {suffix}")
            stem = re.sub(r"[^\w-]", "_", source.stem)[:60]
            name = f"{stem}-{record.image.id[:20]}{suffix}"
            key = Path(name).stem.casefold()
            if key in names:
                raise ValueError("Export filename collision")
            names.add(key)
        if format == 'csv' and not copy_images:
            name = None
        items.append((record, candidate, name))
    files = []
    if not items:
        if not allow_empty:
            reasons = '; '.join(str(item['reason']) for item in skipped[:5])
            raise ValueError(f"No eligible candidates for profile {profile_id!r}; nothing exported. {reasons}")
    else:
        with PUBLISH_LOCK:
            if not beside:
                _safe_path(destination, directory=True)
            if beside:
                files = _sidecar_export(items, overwrite, version, reserved)
            else:
                destination, files = _separate_export(items, destination, format, copy_images, overwrite, version, reserved, allow_empty)
    report = {"format": format, "destination": "Beside original images" if beside else str(destination),
              "exported": len(items), "skipped": skipped, "files": files,
              "semantics": 'record_archive' if format == 'native' else 'selected_training_captions'}
    if format == 'native':
        report.update(scope=('All candidates and analyses are archived, including failed and rejected results.' if full_archive else
                             'Native v2: selected images retain all candidates and analyses. Native v1 retains the selected candidate.'),
                      archived_candidates=sum(len(r.candidates) if not c or r.schema_version == 2 else 1 for r, c, _ in items),
                      archived_analyses=sum(len(r.analyses) for r, _, _ in items))
    return report


def _skip_reason(candidate, record, format, approved_only):
    if not candidate:
        return "No matching caption style or type"
    if approved_only and candidate.approval != "approved":
        return candidate.error or "Caption is not approved"
    if format != 'native':
        if candidate.status != "success" or candidate.approval == "rejected":
            return candidate.error or "Caption was rejected"
        issue = source_issue(candidate, record.candidates, record.analyses)
        if issue:
            return "Unsafe derived caption: " + issue
    return None
