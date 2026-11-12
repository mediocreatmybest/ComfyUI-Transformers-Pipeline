from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from captioning.records import CaptionRecord, ImageIdentity, CaptionCandidate, ModelInfo, AnalysisResult
from captioning.filtering import filter_records
from captioning.export import export_records
from test_import import load_extension, ROOT


class CaptionFilterTests(unittest.TestCase):
    def record(self, score=.5, version=2):
        candidate = CaptionCandidate('caption', 'factual', 'facts', '', {}, ModelInfo('fixture', None, 'image-to-text'), 'success', text='A bird.')
        analysis = AnalysisResult('score', 'classification', 'medium', 'zero-shot-image-classification', None, {}, 'success',
                                  {'labels': ['photograph', 'painting'], 'scores': [score, .9]})
        metadata = AnalysisResult('metadata', 'metadata', 'exif', 'source-metadata', None, {}, 'success',
                                  {'ISO': {'value': '200', 'source': 'fixture'}})
        return CaptionRecord(ImageIdentity('a'*64, 'bird.png', 'b'*64, 10, 10), [candidate], version,
                             [analysis, metadata] if version == 2 else [])

    def test_inclusive_minimum_exact_label_and_highest_score(self):
        for score, expected in ((.49, 1), (.5, 0), (.51, 0), (0, 1)):
            result = filter_records([self.record(score)], 'medium', 'photograph', .5)
            self.assertEqual(len(result[expected]), 1)
        self.assertEqual(len(filter_records([self.record(.1)], 'medium', '', .9)[0]), 1)

    def test_no_score_choices_and_v1(self):
        for reason in ('v1', 'group', 'label', 'failure'):
            record = self.record(version=1 if reason == 'v1' else 2)
            if reason == 'failure':
                record.analyses[0].status = 'failure'; record.analyses[0].values = {}; record.analyses[0].error = 'No result'
            for choice, bucket in (('Keep anyway', 0), ('Leave out', 1), ('Set aside to look at', 2)):
                result = filter_records([record], 'absent' if reason == 'group' else 'medium',
                                        'absent' if reason == 'label' else '', .5, choice)
                self.assertEqual(len(result[bucket]), 1)

    def test_ambiguous_and_stale_scores_are_always_set_aside(self):
        for reason in ('duplicate', 'edited', 'rejected'):
            record = self.record()
            if reason == 'duplicate':
                extra = deepcopy(record.analyses[0]); extra.id = 'second'; record.analyses.append(extra)
            else:
                record.analyses[0].source_candidate_id = 'caption'
                record.analyses[0].settings['source_text'] = 'A bird.'
                if reason == 'edited': record.candidates[0].edit('A tree.')
                else: record.candidates[0].review('rejected')
            self.assertEqual(len(filter_records([record], 'medium', '', 0, 'Keep anyway')[2]), 1)

    def test_originals_and_complete_history_preserved_and_summary_bounded(self):
        record = self.record(); before = record.to_dict()
        outputs = filter_records([record]*100, 'medium', '', .5)
        self.assertEqual(record.to_dict(), before)
        self.assertEqual(outputs[0][0].to_dict(), before)
        outputs[0][0].candidates[0].edit('Changed output')
        self.assertEqual(record.to_dict(), before)
        self.assertLess(len(outputs[3]), 5000)
        self.assertEqual(filter_records([], 'medium')[:3], ([], [], []))

    def test_node_preview_export_and_example_connections(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'bird.png'; Image.new('RGB',(10,10)).save(path)
            record = self.record(); record.image.source_path = str(path)
            record.image.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            original = path.read_bytes()
            nodes = load_extension().NODE_CLASS_MAPPINGS
            result = nodes['CaptionFilterTpl']().filter([record.to_dict()], 'medium', 'photograph', .5, 'Leave out')
            kept = result['result'][0]
            self.assertIn('A bird.', nodes['CaptionPreviewTpl']().preview(caption_records=kept, show_images=False)['result'][0])
            export_records([CaptionRecord.from_dict(r) for r in kept], Path(folder)/'export', 'txt', profile_id='facts')
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(next((Path(folder)/'export').glob('*.txt')).read_text().strip(), 'A bird.')
        prompt = json.loads((ROOT/'example_workflows/api/Caption_Filter.json').read_text())
        filters = [(key,n) for key,n in prompt.items() if n['class_type']=='CaptionFilterTpl']
        self.assertEqual(len(filters),1)
        key = filters[0][0]
        self.assertEqual({n['inputs']['caption_records'][1] for n in prompt.values()
                          if n['class_type']=='CaptionPreviewTpl' and n['inputs'].get('caption_records',[None])[0]==key}, {0,1,2})

    def test_invalid_settings_fail_clearly(self):
        for score in (-1, 2, float('nan')):
            with self.assertRaises(ValueError): filter_records([], 'medium', minimum_score=score)

    def test_changed_ancestor_of_text_classification_needs_a_look(self):
        from captioning.composition import compose_records
        record = compose_records([self.record()], '{caption}', profile_id='derived')[0]
        record.analyses[0].source_candidate_id = record.candidates[-1].id
        record.analyses[0].settings['source_text'] = record.candidates[-1].text
        record.candidates[0].edit('A different subject.')
        self.assertEqual(len(filter_records([record], 'medium', '', 0, 'Keep anyway')[2]), 1)
