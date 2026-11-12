import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from PIL import Image
from captioning.records import CaptionRecord


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.png'
        Image.new('RGB', (16, 12), 'red').save(self.path)

    def pipe(self, task, result):
        class Pipe:
            def __call__(self, *args, **kwargs):
                return result
        pipe = Pipe()
        pipe.task = task
        pipe.model = SimpleNamespace(name_or_path='test/model', config=SimpleNamespace(_commit_hash='test'))
        return pipe

    def test_image_classification_preserves_scores_and_failed_image_positions(self):
        from captioning.classification import classify_images
        pipe = self.pipe('zero-shot-image-classification', [{'label': 'painting', 'score': .73}, {'label': 'photo', 'score': .27}])
        records = classify_images(pipe, [self.path, self.path.with_name('missing.png')], ['photo', 'painting'], 'medium')
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].analyses[0].values['scores'], [.73, .27])
        self.assertEqual(records[1].analyses[0].status, 'failure')
        self.assertNotEqual(records[0].image.id, records[1].image.id)
        self.assertEqual(CaptionRecord.from_dict(records[0].to_dict()).to_dict(), records[0].to_dict())

    def test_malformed_scores_fail_without_empty_success_or_label_loss(self):
        from captioning.classification import classify_images
        for output in [[{'label': 'photo', 'score': float('nan')}], [{'label': 'other', 'score': .9}], []]:
            result = classify_images(self.pipe('zero-shot-image-classification', output), [self.path], ['photo', 'painting'], 'medium')
            self.assertEqual(result[0].analyses[0].status, 'failure')
            self.assertEqual(result[0].analyses[0].values, {})

    def test_text_classification_keeps_source_candidate_and_original(self):
        from captioning.classification import classify_text
        from captioning.generation import generate_records
        from captioning.profiles import CaptionProfile
        pipe = self.pipe('image-text-to-text', None)
        original = generate_records(pipe, [self.path], [CaptionProfile('facts', 'factual', 'describe')], infer=lambda *a: 'A bird.')
        snapshot = original[0].to_dict()
        classifier = self.pipe('zero-shot-classification', {'labels': ['wildlife', 'building'], 'scores': [.8, .2]})
        result = classify_text(classifier, original, ['wildlife', 'building'], 'subject', 'facts')
        self.assertEqual(result[0].analyses[0].source_candidate_id, original[0].candidates[0].id)
        self.assertEqual(original[0].to_dict(), snapshot)
        self.assertEqual(snapshot['schema_version'], 1)
        self.assertNotIn('analyses', snapshot)

    def test_invalid_labels_and_task_are_configuration_errors(self):
        from captioning.classification import classify_images
        with self.assertRaises(ValueError):
            classify_images(self.pipe('text-generation', []), [self.path], ['one', 'two'], 'group')
        for labels in [[], ['same', 'same'], ['']]:
            with self.assertRaises(ValueError):
                classify_images(self.pipe('zero-shot-image-classification', []), [self.path], labels, 'group')


if __name__ == '__main__':
    unittest.main()
