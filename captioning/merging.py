from dataclasses import asdict
import os
from pathlib import Path
from .records import CaptionRecord


def merge_records(*branches):
    """Union branches by image identity; conflicting IDs never overwrite data."""
    merged, paths = {}, {}
    for branch in branches:
        for original in branch or []:
            record = CaptionRecord.from_dict(original.to_dict())
            image = record.image
            path = os.path.normcase(str(Path(image.source_path).resolve()))
            if path in paths and paths[path] != image.id:
                raise ValueError(f'Incompatible image identity at {image.source_path}; source contents changed between branches')
            paths[path] = image.id
            if image.id not in merged:
                merged[image.id] = record
                continue
            target = merged[image.id]
            for key in ('source_path', 'sha256', 'width', 'height'):
                if key in ('width', 'height'):
                    if getattr(target.image, key) is None:
                        setattr(target.image, key, getattr(image, key))
                        continue
                    if getattr(image, key) is None:
                        continue
                if getattr(target.image, key) != getattr(image, key):
                    raise ValueError(f'Incompatible image identity field {key}: {image.id}')
            for key in ('author', 'author_source'):
                left, right = getattr(target.image, key), getattr(image, key)
                if left and right and left != right:
                    raise ValueError(f'Conflicting image {key}: {image.id}')
                if not left:
                    setattr(target.image, key, right)
            for collection in ('candidates', 'analyses'):
                items = getattr(target, collection)
                existing = {item.id: item for item in items}
                for item in getattr(record, collection):
                    if item.id in existing:
                        if asdict(existing[item.id]) != asdict(item):
                            raise ValueError(f'Conflicting {collection} id {item.id}; reconcile edits explicitly')
                    else:
                        items.append(item)
                        existing[item.id] = item
            target.schema_version = max(target.schema_version, record.schema_version)
    return list(merged.values())
