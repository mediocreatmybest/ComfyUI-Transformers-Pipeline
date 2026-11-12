from copy import deepcopy
import uuid
from .generation import runtime_versions
from .identity import image_record
from .records import AnalysisResult, CaptionRecord, ModelInfo


def model_info(pipe):
    provenance = getattr(pipe, '_tpl_model_info', {})
    return ModelInfo(provenance.get('model_id') or pipe.model.name_or_path,
                     provenance.get('revision') or getattr(pipe.model.config, '_commit_hash', None),
                     pipe.task, runtime_versions())


def validate(pipe, labels, group, task, options):
    if pipe.task != task:
        raise ValueError(f'This node requires a {task} pipeline; received {pipe.task}')
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x.strip() for x in labels) or len(set(labels)) != len(labels):
        raise ValueError('Candidate labels must be a nonempty JSON array of distinct strings')
    if not group.strip() or not isinstance(options, dict):
        raise ValueError('Choose a group identifier and JSON object pipeline options')
    if {'candidate_labels', 'images', 'image', 'sequences', 'text', 'inputs'} & options.keys():
        raise ValueError('Use dedicated source and label inputs')


def classify_one(pipe, source, labels, group, options, source_id=None, error=None):
    params = dict(id=uuid.uuid4().hex, kind='classification', group_id=group, task=pipe.task,
                  model=model_info(pipe), settings={'candidate_labels': deepcopy(labels), 'pipeline_kwargs': deepcopy(options),
                  'score_semantics': 'model scores; not calibrated probabilities'}, source_candidate_id=source_id)
    try:
        if error:
            raise ValueError(error)
        output = pipe(source, candidate_labels=labels, **deepcopy(options))
        if isinstance(output, list):
            values = {'labels': [item['label'] for item in output], 'scores': [item['score'] for item in output]}
        else:
            values = {'labels': output['labels'], 'scores': output['scores']}
        if set(values['labels']) != set(labels):
            raise ValueError('Pipeline labels do not match the requested label set')
        return AnalysisResult(**params, status='success', values=values)
    except Exception as exc:
        return AnalysisResult(**params, status='failure', error=f'{type(exc).__name__}: {exc}')


def classify_images(pipe, paths, labels, group, options=None):
    options = options or {}
    validate(pipe, labels, group, 'zero-shot-image-classification', options)
    records = []
    for path in paths:
        record, image, error = image_record(path)
        try:
            record.schema_version = 2
            record.analyses.append(classify_one(pipe, image, labels, group, options, error=error))
            records.append(record)
        finally:
            if image:
                image.close()
    return records


def classify_text(pipe, records, labels, group, source_profile='', options=None):
    options = options or {}
    validate(pipe, labels, group, 'zero-shot-classification', options)
    results = []
    for record in records:
        record = CaptionRecord.from_dict(record.to_dict())
        record.schema_version = 2
        sources = [c for c in record.candidates if not source_profile or c.profile_id == source_profile]
        if not sources:
            record.analyses.append(classify_one(pipe, '', labels, group, options, error='No matching source captions'))
        for candidate in sources:
            record.analyses.append(classify_one(pipe, candidate.text, labels, group, options, candidate.id,
                                  candidate.error if candidate.status == 'failure' else None))
        results.append(record)
    return results
