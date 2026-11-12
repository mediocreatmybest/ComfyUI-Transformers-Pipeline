"""Image inspection uses the same path/content identity as caption generation."""
import hashlib
import io
import os
from pathlib import Path
from PIL import Image, ImageOps
from .records import CaptionRecord, ImageIdentity


def image_record(source):
    path = Path(source).resolve()
    digest = image = author = author_source = error = None
    try:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        with Image.open(io.BytesIO(content)) as original:
            author = original.getexif().get(315) or original.info.get('Artist') or original.info.get('Author')
            if author:
                author, author_source = str(author).strip(), 'source_metadata'
            image = ImageOps.exif_transpose(original).convert('RGB')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    identity = hashlib.sha256((os.path.normcase(str(path)) + '\0' + (digest or '')).encode('utf-8')).hexdigest()
    return CaptionRecord(ImageIdentity(identity, str(path), digest, image.width if image else None,
                         image.height if image else None, author, author_source), []), image, error
