from itertools import islice
import hashlib
from pathlib import Path
import uuid

from ..captioning.records import CaptionRecord, iter_records
from ..captioning.provenance import analysis_source_issue, caption_content, is_derived, source_issue
from .ui_feedback import feedback


def clipped(text, limit=1600):
    text = str(text)
    return text if len(text) <= limit else text[:limit] + ' … [truncated for preview]'


class CaptionPreviewTpl:
    DESCRIPTION = "Read-only preview of connected records, saved JSONL, or image paths with optional legacy captions. Saved-record mode runs no inference. Requeueing a live graph can rerun upstream generation or hit checkpoint protection; use records_file to review without regenerating. Thumbnails are temporary copies; canonical records and source images are unchanged."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}, "optional": {
            "caption_records": ("LIST", {"tooltip": "Connect Caption Records, or use records_file instead."}),
            "image_paths": ("LIST", {"tooltip": "Connect Image Loader for image-only preview or alongside legacy captions."}),
            "captions": ("LIST", {"tooltip": "Optional Caption Generator output; image_paths must have exactly the same length."}),
            "records_file": ("STRING", {"default": "", "tooltip": "Saved native JSONL; no model or inference required. Use only one source mode."}),
            "profile_id": ("STRING", {"default": "", "tooltip": "Blank shows all profiles; otherwise match an exact profile ID. Does not modify records."}),
            "offset": ("INT", {"default": 0, "min": 0, "tooltip": "Zero-based image/record offset. Increase for the next page. Use saved records_file to avoid requeueing upstream generation."}),
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
        lines = [f'Preview: {len(page)} images, offset {offset}; profile {profile_id or "all"}.' +
                 (' More images available; increase offset.' if more else ' End of input.')]
        images = []
        for number, (path, candidates, caption, expected_hash, analyses) in enumerate(page, offset + 1):
            label = f'{number}. {Path(path).name}'
            lines.extend(['', clipped(label), f'Source: {clipped(path, 600)}'])
            if candidates is None:
                lines.append(clipped(caption) if caption is not None else 'Image only; no caption supplied.')
            else:
                selected = [c for c in candidates if not profile_id or c.profile_id == profile_id]
                if not selected:
                    lines.append('No candidates match this profile. Available: ' + clipped(', '.join(c.profile_id for c in candidates)))
                for candidate in selected[:6]:
                    lines.append(clipped(f'[{candidate.profile_id} / {candidate.caption_type}] {candidate.status}; '
                                         f'{candidate.approval}; edited={candidate.edited}; id={candidate.id}'))
                    lines.append(clipped(caption_content(candidate) if candidate.status == 'success' else candidate.error))
                    if is_derived(candidate):
                        transform = candidate.generation_parameters.get('transformation', {})
                        if not isinstance(transform, dict):
                            transform = {}
                        lines.append('Derived caption; factual verification: ' + clipped(transform.get('verification', 'unverified')) +
                                     '. Review status is separate from caption text.')
                        lines.append(clipped('Sources: candidates ' + str(transform.get('source_candidate_ids', [])) +
                                             '; analyses ' + str(transform.get('source_analysis_ids', []))))
                    if candidate.status == 'success':
                        issue = source_issue(candidate, candidates, analyses)
                        if issue:
                            lines.append('Unsafe to reuse/export as training text: ' + clipped(issue))
                if len(selected) > 6:
                    lines.append('Additional candidates omitted; select a profile or inspect the saved records.')
            if show_images:
                try:
                    images.append(self.thumbnail(path, label, expected_hash))
                except Exception as exc:
                    lines.append(f'Image preview unavailable: {clipped(exc, 400)}')
            for analysis in analyses[:12]:
                lines.append(clipped(f'[{analysis.kind}: {analysis.group_id}] {analysis.status}; id={analysis.id}'))
                issue = analysis_source_issue(analysis, candidates, analyses)
                if issue:
                    lines.append('Unsafe analysis source: ' + clipped(issue))
                if analysis.kind == 'classification' and analysis.status == 'success':
                    lines.append(clipped(', '.join(f'{label}: {score:.6g}' for label, score in
                        zip(analysis.values['labels'], analysis.values['scores'])) + ' (model scores, not calibrated probabilities)'))
                else:
                    lines.append(clipped(analysis.error or str(analysis.values)))
            if len(analyses) > 12:
                lines.append('Additional analyses omitted from preview; preserved in records.')
        text = clipped('\n'.join(lines), 23000)
        result = feedback((text,), text)
        result['ui']['tpl_images'] = images
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
