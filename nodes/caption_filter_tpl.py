from ..captioning.filtering import filter_records, NO_SCORE_CHOICES
from ..captioning.records import CaptionRecord
from .ui_feedback import feedback


class CaptionFilterTpl:
    DESCRIPTION = 'Choose whole images and their captions using scores already saved in your records. No model runs here. Scores are model suggestions, not guarantees. All caption history stays with each image.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'caption_records': ('LIST', {'tooltip': 'Connect records containing captions and classification results.'}),
            'group_id': ('STRING', {'default': 'medium', 'tooltip': 'Which group of labels? Enter the existing group ID, such as medium or subject.'}),
            'label': ('STRING', {'default': '', 'tooltip': 'Keep images matching this exact label, such as photograph. Leave blank to use the highest score in the group.'}),
            'minimum_score': ('FLOAT', {'default': .5, 'min': 0, 'max': 1, 'step': .01, 'display': 'slider', 'tooltip': 'Minimum Score: keep an image when its model score reaches this value or higher.'}),
            'no_score': (list(NO_SCORE_CHOICES), {'default': 'Set aside to look at', 'tooltip': 'No Score Available: what should happen if this group or label has no usable score? Mixed results or changed source captions always need a look.'}),
        }}

    RETURN_TYPES = ('LIST', 'LIST', 'LIST', 'STRING')
    RETURN_NAMES = ('Kept', 'Left Out', 'Needs a Look', 'What Happened')
    FUNCTION = 'filter'
    CATEGORY = 'Transformers Pipeline/Datasets'
    OUTPUT_NODE = True

    def filter(self, caption_records, group_id, label='', minimum_score=.5, no_score='Set aside to look at'):
        kept, left, look, summary = filter_records([CaptionRecord.from_dict(r) for r in caption_records],
                                                   group_id, label, minimum_score, no_score)
        return feedback(tuple([r.to_dict() for r in bucket] for bucket in (kept, left, look)) + (summary,), summary)
