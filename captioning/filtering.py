"""Partition complete records using existing scores; no inference or file writes."""
from pathlib import Path
import math
from .records import CaptionRecord
from .provenance import analysis_source_issue


NO_SCORE_CHOICES = ('Keep anyway', 'Leave out', 'Set aside to look at')


def filter_records(records, group_id, label='', minimum_score=.5, no_score='Set aside to look at'):
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError('Enter a label group, such as medium or subject.')
    if type(minimum_score) not in (int, float) or not math.isfinite(minimum_score) or not 0 <= minimum_score <= 1:
        raise ValueError('Minimum Score must be between 0 and 1.')
    if no_score not in NO_SCORE_CHOICES or not isinstance(label, str):
        raise ValueError('Choose a No Score Available option and an exact label or leave the label blank.')
    buckets = ([], [], [])
    details = []
    for original in records:
        record = CaptionRecord.from_dict(original.to_dict())
        matches = [a for a in record.analyses if a.kind == 'classification' and a.group_id == group_id]
        bucket, reason = NO_SCORE_CHOICES.index(no_score), 'No score available for this group.'
        if len(matches) > 1:
            bucket, reason = 2, 'More than one result in this group; choose a single result before filtering.'
        elif matches:
            analysis = matches[0]
            issue = analysis_source_issue(analysis, record.candidates, record.analyses)
            if issue:
                bucket, reason = 2, 'The source caption has changed, was rejected, or needs checking. Run its classification again.'
            elif analysis.status != 'success':
                reason = 'No score available: the model did not return a result.'
            else:
                scores = dict(zip(analysis.values['labels'], analysis.values['scores']))
                chosen = label if label else max(scores, key=scores.get)
                if chosen not in scores:
                    reason = 'No score available for the chosen label.'
                else:
                    score = scores[chosen]
                    bucket = 0 if score >= minimum_score else 1
                    reason = f'{chosen[:100]}: {score:g}; minimum {minimum_score:g}.'
        buckets[bucket].append(record)
        if len(details) < 12:
            filename = Path(record.image.source_path).name[:120]
            details.append(f'{filename} — {("Keep", "Leave Out", "Needs a Look")[bucket]}: {reason}')
    total = sum(map(len, buckets))
    summary = f'Images Kept: {len(buckets[0])} | Images Left Out: {len(buckets[1])} | Needs a Look: {len(buckets[2])}'
    if total > len(details):
        details.append(f'{total-len(details)} more images; connect a Preview to see them.')
    return (*buckets, summary + ('\n' + '\n'.join(details) if details else ''))
