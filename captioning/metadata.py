import hashlib
import io
from pathlib import Path
import uuid
from PIL import Image
from .identity import image_record
from .records import AnalysisResult


EXIF_FIELDS = {271: 'Camera Make', 272: 'Camera Model', 315: 'Artist', 33434: 'Exposure Time',
               33437: 'F Number', 34855: 'ISO', 37386: 'Focal Length', 42036: 'Lens Model', 36867: 'Date Taken'}


def metadata_records(paths):
    records = []
    for path in paths:
        record, _, error = image_record(path, decode=False)
        record.schema_version = 2
        params = dict(id=uuid.uuid4().hex, kind='metadata', group_id='exif', task='source-metadata',
                      model=None, settings={'reader': 'Pillow EXIF', 'fields': list(EXIF_FIELDS.values())})
        try:
            if error:
                raise ValueError(error)
            content = Path(path).read_bytes()
            if hashlib.sha256(content).hexdigest() != record.image.sha256:
                raise ValueError('Source changed during metadata extraction')
            with Image.open(io.BytesIO(content)) as image:
                tags = image.getexif()
                fields = dict(tags)
                if 34665 in tags:
                    fields.update(tags.get_ifd(34665))
                values = {}
                for tag, name in EXIF_FIELDS.items():
                    if tag in fields:
                        value = fields[tag]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='replace').rstrip('\0')
                        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(x, int) for x in value):
                            value = f'{value[0]}/{value[1]}'
                        values[name] = {'value': str(value), 'source': f'EXIF:{tag}'}
                result = AnalysisResult(**params, status='success', values=values)
        except Exception as exc:
            result = AnalysisResult(**params, status='failure', error=f'{type(exc).__name__}: {exc}')
        record.analyses.append(result)
        records.append(record)
    return records
