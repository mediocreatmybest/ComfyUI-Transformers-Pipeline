from copy import deepcopy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from captioning.generation import generate_records
from captioning.profiles import CaptionProfile
from captioning.records import write_records
from test_import import load_extension


class UsabilityTests(unittest.TestCase):
    def setUp(self):
        self.extension = load_extension()
        self.nodes = self.extension.NODE_CLASS_MAPPINGS
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'café.png'
        Image.new('RGB', (900, 600), 'red').save(self.image)
        self.pipe = SimpleNamespace(task='image-text-to-text', model=SimpleNamespace(
            name_or_path='org/model', config=SimpleNamespace(model_type='test', _commit_hash='abc')))
        self.records = generate_records(self.pipe, [self.image, self.root / 'missing.png'],
            [CaptionProfile('facts', 'factual', 'describe'), CaptionProfile('style', 'style', 'describe')],
            infer=lambda *args: 'A red study.')

    def test_preview_shows_candidates_failures_profiles_and_keeps_records_unchanged(self):
        node = self.nodes['CaptionPreviewTpl']()
        records = [r.to_dict() for r in self.records]
        before = deepcopy(records)
        result = node.preview(caption_records=records, show_images=False)
        text = result['result'][0]
        for expected in ['café.png', 'missing.png', 'facts', 'style', 'failure', 'A red study.', 'unreviewed']:
            self.assertIn(expected, text)
        self.assertEqual(result['ui']['text'], [text])
        self.assertEqual(records, before)
        self.assertEqual(result['ui']['tpl_images'], [])

    def test_saved_preview_pages_without_inference_or_full_file_loading(self):
        checkpoint = self.root / 'records.jsonl'
        write_records(self.records, checkpoint)
        with checkpoint.open('a', encoding='utf-8') as stream:
            stream.write('invalid later record\n')
        with patch('tpl.captioning.generation.generate_records', side_effect=AssertionError('inference')):
            node = self.nodes['CaptionPreviewTpl']()
            text = node.preview(records_file=str(checkpoint), profile_id='facts', limit=1, show_images=False)['result'][0]
        self.assertIn('A red study.', text)
        self.assertNotIn('[style', text)
        self.assertIn('more', text.lower())
        self.assertEqual(checkpoint.read_text(encoding='utf-8').splitlines()[-1], 'invalid later record')

    def test_preview_bounds_text_and_only_decodes_selected_images(self):
        records = [r.to_dict() for r in self.records]
        records[0]['candidates'][0]['text'] = 'x' * 100000
        original = self.image.read_bytes()
        with patch.dict('sys.modules', {'folder_paths': SimpleNamespace(get_temp_directory=lambda: str(self.root / 'temp'))}):
            result = self.nodes['CaptionPreviewTpl']().preview(caption_records=records, limit=1, show_images=True)
        self.assertLess(len(result['result'][0]), 25000)
        self.assertIn('truncated', result['result'][0])
        images = result['ui']['tpl_images']
        self.assertEqual(len(images), 1)
        self.assertIn('café.png', images[0]['label'])
        with Image.open(self.root / 'temp' / images[0]['subfolder'] / images[0]['filename']) as thumb:
            self.assertLessEqual(max(thumb.size), 512)
        self.assertEqual(self.image.read_bytes(), original)

    def test_preview_legacy_alignment_image_only_and_missing_source_feedback(self):
        node = self.nodes['CaptionPreviewTpl']()
        with self.assertRaisesRegex(ValueError, 'count'):
            node.preview(captions=['one'], image_paths=[str(self.image), 'missing'])
        result = node.preview(captions=['café'], image_paths=[str(self.image)], show_images=False)
        self.assertIn('café', result['result'][0])
        result = node.preview(image_paths=['missing.png'], show_images=True)
        self.assertIn('preview unavailable', result['result'][0].lower())
        with self.assertRaisesRegex(ValueError, 'one source'):
            node.preview(caption_records=[], records_file='records.jsonl', show_images=False)

    def test_saved_caption_does_not_show_a_replaced_source_as_its_original(self):
        Image.new('RGB', (900, 600), 'blue').save(self.image)
        with patch.dict('sys.modules', {'folder_paths': SimpleNamespace(get_temp_directory=lambda: str(self.root / 'temp'))}):
            result = self.nodes['CaptionPreviewTpl']().preview(caption_records=[self.records[0].to_dict()], show_images=True)
        self.assertEqual(result['ui']['tpl_images'], [])
        self.assertIn('changed since captioning', result['result'][0])
        self.assertIn('A red study.', result['result'][0])

    def test_exif_keeps_empty_and_failed_positions_and_continues(self):
        paths = [str(self.image), str(self.root / 'absent.png'), str(self.image)]
        fake = SimpleNamespace(process_file=lambda stream: {})
        with patch.dict('sys.modules', {'exifread': fake}):
            metadata, formatted = self.nodes['ExifMetadataExtractorTpl']().extract_exif_metadata(paths)
        self.assertEqual(len(metadata), 3)
        self.assertEqual(len(formatted), 3)
        self.assertEqual(metadata[0], {})
        self.assertEqual(formatted[0], '')
        self.assertIn('Error', metadata[1])
        self.assertIn('absent.png', formatted[1])
        self.assertEqual(metadata[2], {})

    def test_image_discovery_feedback_does_not_decode_files(self):
        (self.root / 'bad.jpg').write_bytes(b'not an image')
        (self.root / 'café.txt').write_text('existing')
        node = self.nodes['ImageLoaderTpl']()
        with patch('PIL.Image.open', side_effect=AssertionError('should not decode')):
            result = getattr(node, node.FUNCTION)(str(self.root), True)
        self.assertEqual(result['result'][0], [str(self.root / 'bad.jpg')])
        self.assertIn('Selected 1', result['ui']['text'][0])
        self.assertIn('Skipped 1', result['ui']['text'][0])
        self.assertIn('bad.jpg', result['ui']['text'][0])

    def test_debug_is_bounded_quiet_and_preserves_passthrough(self):
        data = ['x' * 100000] * 20
        with contextlib.redirect_stdout(io.StringIO()) as output:
            raw, text = self.nodes['DebugNodeTpl']().debug_data(data)
            model, description = self.nodes['DebugModelNodeTpl']().debug_pipeline(self.pipe)
        self.assertIs(raw, data)
        self.assertIs(model, self.pipe)
        self.assertLess(len(text), 5000)
        self.assertLess(len(description), 5000)
        self.assertEqual(output.getvalue(), '')

    def test_record_protection_feedback_does_not_repeat_inference(self):
        checkpoint = self.root / 'records.jsonl'
        write_records(self.records, checkpoint)
        original = checkpoint.read_bytes()
        node = self.nodes['CaptionRecordsTpl']()
        with patch('tpl.nodes.caption_records_tpl.generate_records', side_effect=AssertionError('inference')):
            with self.assertRaisesRegex(FileExistsError, 'Preview|preview'):
                node.generate(self.pipe, [str(self.image)], records_path=str(checkpoint))
        self.assertEqual(checkpoint.read_bytes(), original)

    def test_export_ui_distinguishes_protection_from_errors(self):
        node = self.nodes['DatasetExportTpl']()
        kwargs = dict(destination=str(self.root / 'dataset'), format='txt', profile_id='facts',
                      caption_records=[r.to_dict() for r in self.records])
        first = getattr(node, node.FUNCTION)(**kwargs)
        self.assertIn('Saved 1', first['ui']['text'][0])
        self.assertIn('Skipped 1', first['ui']['text'][0])
        second = getattr(node, node.FUNCTION)(**kwargs)
        self.assertIn('Protected', second['ui']['text'][0])
        self.assertEqual(json.loads(second['result'][0])['status'], 'protected')
        with self.assertRaises(ValueError):
            getattr(node, node.FUNCTION)(**{**kwargs, 'profile_id': 'unknown'})


if __name__ == '__main__':
    unittest.main()
