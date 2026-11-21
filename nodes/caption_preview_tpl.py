from fractions import Fraction
from itertools import islice
import hashlib
from pathlib import Path
import re
import uuid

from ..captioning.records import CaptionRecord, iter_records
from ..captioning.provenance import analysis_source_issue, caption_content, is_derived, source_issue
from .ui_feedback import feedback


def clipped(text, limit=1600):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + ' … [truncated for preview]'


def friendly_name(identifier):
    return clipped(identifier.replace('_', ' ').replace('-', ' ').title(), 100)


def warning_text(value):
    return clipped(re.sub(r'\b(?:[0-9a-f]{64}|[0-9a-f]{32})\b', '[record ID]', str(value), flags=re.I), 400)


def camera_summary(fields):
    values = {name: str(value.get('value', '')).strip() for name, value in fields.items()
              if isinstance(value, dict) and value.get('value') is not None}
    camera = values.get('Camera Model') or values.get('Camera Make')
    parts = [camera, values.get('Lens Model')]
    for name, pattern in (('Focal Length', '{} mm'), ('F Number', 'f/{}'),
                          ('Exposure Time', '{} s'), ('ISO', 'ISO {}')):
        try:
            number = Fraction(values.get(name, ''))
            if number <= 0:
                continue
            display = str(number) if name == 'Exposure Time' and number < 1 else f'{float(number):g}'
            parts.append(pattern.format(display))
        except (ValueError, ZeroDivisionError, OverflowError):
            continue
    date = values.get('Date Taken', '')
    if date and not date.startswith('0000:00:00'):
        parts.append(re.sub(r'^(\d{4}):(\d{2}):(\d{2})', r'\1-\2-\3', date))
    return clipped(' · '.join(part for part in parts if part), 600)


def preview_card(path, candidates, caption, analyses, profile_id):
    card = {'filename': clipped(Path(path).name, 240), 'captions': [], 'classifications': [], 'warnings': []}
    details = [f'Source: {clipped(path, 600)}']
    if candidates is None:
        if caption and str(caption).strip():
            card['captions'].append({'label': 'Caption', 'text': clipped(caption)})
    else:
        original_id = next((c.id for c in candidates if not is_derived(c)), None)
        selected = [c for c in candidates if not profile_id or c.profile_id == profile_id]
        selected.sort(key=is_derived)  # Show the original models before downstream transformations.
        if not selected and candidates:
            available = ', '.join(dict.fromkeys(friendly_name(c.profile_id) for c in candidates))
            card['warnings'].append('No captions match this style. Available: ' + clipped(available))
        elif not selected:
            card['warnings'].append('No captions in this record.')
        for candidate in selected[:6]:
            profile = friendly_name(candidate.profile_id)
            kind = ('Edited' if candidate.edited else 'Composed' if candidate.caption_type == 'composed'
                    else 'Synthesized' if is_derived(candidate) else 'Original' if candidate.id == original_id else 'Generated alternative')
            label = f'{profile} · {kind}'
            details.append(clipped(f'[{candidate.profile_id} / {candidate.caption_type}] {candidate.status}; '
                                   f'{candidate.approval}; id={candidate.id}; '
                                   f'model={candidate.model.model_id}; revision={candidate.model.revision or "unknown"}'))
            if candidate.status == 'failure':
                card['warnings'].append(f'{label}: {warning_text(candidate.error)}')
                continue
            item = {'label': label, 'text': clipped(caption_content(candidate))}
            if candidate.tags:
                item['tags'] = [clipped(tag, 100) for tag in candidate.tags[:20]]
            card['captions'].append(item)
            if candidate.edited and candidate.generated_text and candidate.generated_text != candidate.text:
                details.append('Original generated text: ' + clipped(candidate.generated_text))
            if candidate.approval == 'rejected':
                card['warnings'].append(f'{label}: Rejected in review.')
            if is_derived(candidate):
                transform = candidate.generation_parameters.get('transformation', {})
                verification = transform.get('verification', 'unverified') if isinstance(transform, dict) else 'unverified'
                if verification == 'unverified':
                    card['warnings'].append(f'{label}: Derived caption, factual accuracy unverified.')
                details.append(clipped('Transformation: ' + str(transform)))
            issue = source_issue(candidate, candidates, analyses)
            if issue and candidate.approval != 'rejected':
                card['warnings'].append(f'{label}: Unsafe to reuse as training text. {issue.split(":", 1)[0]}.')
                details.append('Training export check: ' + clipped(issue))
        if len(selected) > 6:
            card['warnings'].append('More caption candidates are available. Select a Caption Style to narrow the preview.')
    exif = {}
    for analysis in analyses[:12]:
        details.append(clipped(f'[{analysis.kind}: {analysis.group_id}] {analysis.status}; id={analysis.id}; '
                               + str(analysis.error or analysis.values), 1000))
        label = friendly_name(analysis.group_id)
        issue = analysis_source_issue(analysis, candidates, analyses)
        if analysis.status == 'failure':
            card['warnings'].append(f'{label}: {warning_text(analysis.error)}')
        elif issue:
            card['warnings'].append(f'{label}: Classification source needs review. {issue.split(":", 1)[0]}.')
            details.append('Analysis source check: ' + clipped(issue))
        elif analysis.kind == 'classification':
            if analysis.source_candidate_id and analysis.source_candidate_id not in {c.id for c in selected[:6]}:
                continue
            name, score = max(zip(analysis.values['labels'], analysis.values['scores']), key=lambda item: item[1])
            card['classifications'].append(f'{label}: {clipped(name, 120)} (model score {score:.3g})')
        elif analysis.kind == 'metadata' and analysis.group_id == 'exif':
            exif.update(analysis.values)
    if len(analyses) > 12:
        details.append('Additional analyses are preserved in the records.')
    camera = camera_summary(exif)
    if camera:
        card['camera'] = camera
    card['details'] = clipped('\n'.join(details), 4800)
    return card


class CaptionPreviewTpl:
    DESCRIPTION = "Read-only preview of connected records, saved JSONL, or image paths with optional legacy captions. Saved-record mode runs no inference. Paging reuses cached upstream results when available. Load a saved records file to review with no model. Thumbnails are temporary copies; canonical records and source images are unchanged."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "caption_records": ("LIST", {"tooltip": "Connect Caption Records, or use records_file instead."}),
            "image_paths": ("LIST", {"tooltip": "Connect Image Loader for image-only preview or alongside legacy captions."}),
            "captions": ("LIST", {"tooltip": "Optional Caption Generator output; image_paths must have exactly the same length."}),
            "records_file": ("STRING", {"default": "", "tooltip": "Saved native JSONL; no model or inference required. Use only one source mode."}),
            "profile_id": ("STRING", {"default": "", "tooltip": "Blank shows all profiles; otherwise match an exact profile ID. Does not modify records."}),
            "offset": ("INT", {"default": 0, "min": 0, "tooltip": "Zero-based image/record offset. Increase for the next page. Queue to load this page. Use saved records_file for offline review."}),
            "limit": ("INT", {"default": 4, "min": 1, "max": 8, "tooltip": "Maximum images on this page. Only these images are decoded for thumbnails."}),
            "show_images": ("BOOLEAN", {"default": True, "tooltip": "Create thumbnails up to 512 pixels in ComfyUI temp storage. Disable for text-only/headless review."}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preview_text",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "Transformers Pipeline/Datasets"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float('nan')  # Saved records and source images can change externally.

    def preview(self, caption_records=None, image_paths=None, captions=None, records_file='',
                profile_id='', offset=0, limit=4, show_images=True):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8:
            raise ValueError('Preview offset must be nonnegative and limit must be between 1 and 8')
        if sum((caption_records is not None, bool(records_file), image_paths is not None)) != 1:
            raise ValueError('Choose exactly one source: caption_records, records_file, or image_paths with optional captions')
        if captions is not None and (image_paths is None or len(captions) != len(image_paths)):
            raise ValueError('Caption/image count mismatch; connect matching image_paths and captions')
        if image_paths is not None:
            rows = ((path, None, captions[i] if captions is not None else None, None, []) for i, path in enumerate(image_paths))
        else:
            records = iter_records(records_file) if records_file else (
                CaptionRecord.from_dict(r.to_dict() if hasattr(r, 'to_dict') else r) for r in caption_records)
            rows = ((r.image.source_path, r.candidates, None, r.image.sha256, r.analyses) for r in records)
        # One lookahead reports another page without reading the rest of a JSONL file.
        try:
            page = list(islice(rows, offset, offset + limit + 1))
        finally:
            rows.close()
            if records_file:
                records.close()
        more = len(page) > limit
        page = page[:limit]
        summary = (f'Preview: {len(page)} images, offset {offset}.'
                   + (' More images available; increase offset.' if more else ' End of input.'))
        lines, images, cards = [summary], [], []
        for number, (path, candidates, caption, expected_hash, analyses) in enumerate(page, offset + 1):
            card = preview_card(path, candidates, caption, analyses, profile_id)
            if show_images:
                try:
                    image = self.thumbnail(path, f"{number}. {card['filename']}", expected_hash)
                    images.append(image)
                    card['image'] = image
                except Exception as exc:
                    card['warnings'].append(f'Image preview unavailable: {clipped(exc, 400)}')
            cards.append(card)
            lines.extend(['', f"{number}. {card['filename']}"])
            if candidates is None and not card['captions']:
                lines.append('Image only; no caption supplied.')
            for item in card['captions']:
                lines.extend([item['label'], item['text']])
            lines.extend(card['classifications'])
            if card.get('camera'):
                lines.append(card['camera'])
            lines.extend(card['warnings'])
            lines.append(card['details'])
        text = clipped('\n'.join(lines), 23000)
        result = feedback((text,), text)
        result['ui'].update(tpl_images=images, tpl_cards=cards, tpl_summary=[summary])
        return result

    @staticmethod
    def thumbnail(path, label, expected_hash=None):
        from PIL import Image, ImageOps
        if not Path(path).is_file():
            raise FileNotFoundError(f'Source image does not exist: {path}')
        with open(path, 'rb') as stream:
            if expected_hash:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
                if digest.hexdigest() != expected_hash:
                    raise ValueError('Source image changed since captioning; thumbnail omitted. Re-caption or restore the original image.')
                stream.seek(0)
            import folder_paths
            folder = Path(folder_paths.get_temp_directory()) / 'transformers_preview'
            folder.mkdir(parents=True, exist_ok=True)
            filename = uuid.uuid4().hex + '.png'
            with Image.open(stream) as source:
                source.draft('RGB', (512, 512))
                source.thumbnail((512, 512))
                with ImageOps.exif_transpose(source) as oriented:
                    oriented.convert('RGB').save(folder / filename)
        return {'filename': filename, 'subfolder': folder.name, 'type': 'temp', 'label': label}
