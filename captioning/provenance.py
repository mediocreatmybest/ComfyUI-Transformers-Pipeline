"""Read-only caption presentation and derivation checks shared by composition/export/preview."""
from dataclasses import asdict


def is_derived(candidate):
    transform = candidate.generation_parameters.get('transformation')
    return candidate.caption_type in {'composed', 'synthetic'} or (
        isinstance(transform, dict) and transform.get('method') in {'template', 'text-generation'})


def caption_content(candidate):
    """Remove only the exact, unedited status prefix written by older synthesis code."""
    transform = candidate.generation_parameters.get('transformation', {})
    if (isinstance(transform, dict) and transform.get('method') == 'text-generation'
            and not candidate.edited and candidate.generated_text
            and candidate.text == 'Draft synthesis (unverified): ' + candidate.generated_text):
        return candidate.generated_text
    return candidate.text


def source_issue(candidate, candidates, analyses):
    """Return why a candidate is unsafe to reuse, including changed ancestors.

    Missing snapshots remain readable/archiveable, but cannot establish freshness.
    Approval alone does not prove factual accuracy or repair stale provenance.
    """
    current = {c.id: c for c in candidates}
    current_analyses = {a.id: a for a in analyses}
    verified = set()  # Shared ancestors need only one check per call, never across edits.

    def visit(candidate_id, trail):
        if candidate_id in verified:
            return None
        source = current.get(candidate_id)
        if source is None or source.status != 'success' or source.approval == 'rejected':
            return f'Unavailable or rejected source candidate: {candidate_id}'
        if candidate_id in trail:
            return 'Cyclic source provenance'
        if not is_derived(source):
            verified.add(candidate_id)
            return None
        transform = source.generation_parameters.get('transformation')
        if not isinstance(transform, dict) or not {'source_candidate_ids', 'source_analysis_ids'} <= transform.keys():
            return f'Unverifiable source provenance: {candidate_id}; regenerate the derivation'
        trail = trail | {candidate_id}
        snapshots = {s['id']: s for s in transform.get('source_candidate_snapshots', []) if isinstance(s, dict) and 'id' in s}
        for source_id in transform['source_candidate_ids']:
            snapshot, original = snapshots.get(source_id), current.get(source_id)
            if snapshot is None or original is None or not {'text', 'tags'} <= snapshot.keys():
                return f'Missing source candidate snapshot: {source_id}; regenerate the derivation'
            if snapshot['text'] != original.text or snapshot['tags'] != original.tags:
                return f'Stale source candidate: {source_id}; regenerate the derivation'
            issue = visit(source_id, trail)
            if issue:
                return issue
        snapshots = {s['id']: s for s in transform.get('source_analysis_snapshots', []) if isinstance(s, dict) and 'id' in s}
        for analysis_id in transform['source_analysis_ids']:
            original = current_analyses.get(analysis_id)
            if original is None or snapshots.get(analysis_id) != asdict(original):
                return f'Stale or missing source analysis snapshot: {analysis_id}; regenerate the derivation'
            if original.status != 'success':
                return f'Failed source analysis: {analysis_id}'
            if original.source_candidate_id:
                source = current.get(original.source_candidate_id)
                if source is None or original.settings.get('source_text') != source.text:
                    return f'Stale text classification source: {analysis_id}; reclassify the source caption'
                issue = visit(source.id, trail)
                if issue:
                    return issue
        verified.add(candidate_id)
        return None

    return visit(candidate.id, set())


def analysis_source_issue(analysis, candidates, analyses):
    if not analysis.source_candidate_id:
        return None
    source = next((c for c in candidates if c.id == analysis.source_candidate_id), None)
    if source is None or analysis.settings.get('source_text') != source.text:
        return 'Stale text classification; reclassify the edited source caption'
    return source_issue(source, candidates, analyses)
