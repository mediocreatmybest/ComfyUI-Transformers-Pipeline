import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from PIL import Image
from captioning.generation import generate_records
from captioning.profiles import CaptionProfile
from captioning.records import read_records
from captioning.classification import classify_images
from captioning.metadata import metadata_records
from captioning.merging import merge_records


class CompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'fixture.jpg'
        exif = Image.Exif(); exif[34855] = 200; exif[33437] = 5.6; exif[33434] = (1, 500)
        Image.new('RGB', (12, 10), 'red').save(self.path, exif=exif)
        self.pipe = SimpleNamespace(task='image-text-to-text', model=SimpleNamespace(name_or_path='test/model', config=SimpleNamespace()))
        self.captions = generate_records(self.pipe, [self.path], [CaptionProfile('facts', 'factual', 'describe')], infer=lambda *a: 'A bird on a fence.')
        self.records = merge_records(self.captions, metadata_records([self.path]))

    def test_template_uses_fixture_metadata_and_preserves_original_and_approval(self):
        from captioning.composition import compose_records
        self.records[0].candidates[0].review('approved')
        old = self.records[0].to_dict()
        results = compose_records(self.records, '{caption} Captured at {metadata}.', selection={'metadata_fields': ['ISO', 'F Number', 'Exposure Time']})
        new = results[0].candidates[-1]
        self.assertIn('ISO 200', new.text)
        self.assertIn('1/500', new.text)
        self.assertEqual(new.approval, 'unreviewed')
        self.assertEqual(new.generation_parameters['transformation']['source_candidate_ids'], [self.captions[0].candidates[0].id])
        self.assertEqual(self.records[0].to_dict(), old)
        self.assertEqual(results[0].candidates[0].approval, 'approved')

    def test_model_synthesis_and_failures_append_candidates(self):
        from captioning.composition import compose_records
        class Generator:
            task = 'text-generation'
            model = SimpleNamespace(name_or_path='test/generator', config=SimpleNamespace())
            tokenizer = SimpleNamespace(chat_template=None)
            def __call__(self, prompt, **kwargs):
                return [{'generated_text': 'A concise bird description.'}]
        result = compose_records(self.records, '{caption}', pipe=Generator())
        self.assertEqual(result[0].candidates[-1].generated_text, 'A concise bird description.')
        self.assertTrue(result[0].candidates[-1].text.startswith('Draft synthesis (unverified):'))
        self.assertEqual(result[0].candidates[-1].model.pipeline_task, 'text-generation')
        failed = compose_records(self.records, '{caption}', selection={'profile_ids': ['missing']})
        self.assertEqual(failed[0].candidates[-1].status, 'failure')
        self.assertEqual(failed[0].candidates[0].text, 'A bird on a fence.')

    def test_classifications_stay_uncertain_and_native_export_keeps_sources(self):
        from captioning.composition import compose_records
        from captioning.export import export_records
        class Classifier:
            task = 'zero-shot-image-classification'
            model = SimpleNamespace(name_or_path='test/classifier', config=SimpleNamespace())
            def __call__(self, *a, **kw):
                return [{'label': 'photograph', 'score': .8}, {'label': 'painting', 'score': .2}]
        records = merge_records(self.records, classify_images(Classifier(), [self.path], ['photograph', 'painting'], 'medium'))
        result = compose_records(records, '{caption} {classifications}')
        self.assertIn('Possible medium', result[0].candidates[-1].text)
        self.assertIn('uncalibrated', result[0].candidates[-1].text)
        export_records(result, self.root / 'native', 'native', profile_id='composed')
        restored = read_records(self.root / 'native/records.jsonl')[0]
        self.assertEqual(restored.to_dict(), result[0].to_dict())

    def test_invalid_template_and_ambiguous_metadata_fail_explicitly(self):
        from captioning.composition import compose_records
        with self.assertRaises(ValueError):
            compose_records(self.records, '{caption.__class__}')
        more = metadata_records([self.path]); more[0].analyses[0].values['ISO']['value'] = '400'
        combined = merge_records(self.records, more)
        result = compose_records(combined, '{metadata}')
        self.assertEqual(result[0].candidates[-1].status, 'failure')


if __name__ == '__main__':
    unittest.main()
