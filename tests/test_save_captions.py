import csv
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from captioning.export import export_records
from captioning.generation import generate_records
from captioning.profiles import CaptionProfile
from captioning.records import AnalysisResult, publish_file, read_records, write_records
from test_import import load_extension


class SaveCaptionsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.sources = [self.root / 'a/photo.png', self.root / 'a/photo.jpg', self.root / 'b/photo.png']
        for source in self.sources:
            source.parent.mkdir(exist_ok=True)
            Image.new('RGB', (5, 5)).save(source)
        pipe = SimpleNamespace(task='image-to-text', model=SimpleNamespace(name_or_path='fake'))
        self.records = generate_records(pipe, self.sources, [CaptionProfile('photo', 'factual', '', prompt_mode='raw')],
                                        infer=lambda *_: 'A photograph.')

    def save(self, format='txt', folder='output', **kwargs):
        return export_records(self.records, self.root / folder, format, **kwargs)

    def test_txt_files_match_copied_images_and_can_omit_copies(self):
        for copy_images in (False, True):
            report = self.save(folder=str(copy_images), copy_images=copy_images)
            files = list(Path(report['destination']).iterdir())
            self.assertEqual(len(files), 6 if copy_images else 3)
            self.assertEqual(len({path.stem for path in files}), 3)
            self.assertTrue(all(path.read_text() == 'A photograph.\n' for path in files if path.suffix == '.txt'))

    def test_imagefolder_always_has_matching_images(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            self.save('imagefolder', copy_images=False)
        report = self.save('imagefolder', copy_images=True)
        import json
        output = Path(report['destination'])
        rows = [json.loads(line) for line in (output / 'metadata.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all((output / row['file_name']).is_file() for row in rows))

    def test_csv_round_trips_text_and_neutralizes_formulas(self):
        self.records[0].candidates[0].edit('Café, "quoted"\nsecond line', ['night, sky'])
        self.records[1].candidates[0].edit('=1+2')
        self.records[2].candidates[0].edit('+SUM(1,2)')
        report = self.save('csv', copy_images=False)
        with (Path(report['destination']) / 'captions.csv').open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['source_filename'], 'photo.png')
        self.assertEqual(rows[0]['caption'], 'Café, "quoted"\nsecond line')
        self.assertEqual(rows[1]['caption'], "'=1+2")
        self.assertEqual(rows[2]['caption'], "'+SUM(1,2)")
        self.assertEqual(rows[0]['tags'], '["night, sky"]')

    def test_sidecars_protection_replace_batch_version_and_source_integrity(self):
        originals = {path: path.read_bytes() for path in self.sources}
        self.save(beside=True, copy_images=False)
        self.records[0].candidates[0].edit('Updated upstream.')
        with self.assertRaises(FileExistsError):
            self.save(beside=True, copy_images=False)
        self.save(beside=True, copy_images=False, overwrite=True)
        self.assertEqual(self.sources[0].with_name('photo.png.txt').read_text(), 'Updated upstream.\n')
        report = self.save(beside=True, copy_images=False, version=True)
        self.assertEqual(len(set(Path(name).name.split('.v-')[1] for name in report['files'])), 1)
        self.assertTrue(all(path.read_bytes() == data for path, data in originals.items()))
        self.assertFalse((self.root / 'output').exists())

    def test_separate_version_and_protection_do_not_leave_partial_exports(self):
        first = self.save(copy_images=False)
        with self.assertRaises(FileExistsError):
            self.save(copy_images=False)
        second = self.save(copy_images=False, version=True)
        self.assertNotEqual(first['destination'], second['destination'])
        self.assertEqual(first['files'], second['files'])
        self.assertEqual(len(list(Path(second['destination']).iterdir())), 3)

    def test_ambiguous_records_require_selection_and_archive_keeps_every_candidate(self):
        from captioning.composition import compose_records
        self.records = compose_records(self.records, template='{caption}', selection={'profile_ids': ['photo']})
        self.records[0].candidates[0].review('approved')
        self.records[0].analyses.append(AnalysisResult('medium', 'classification', 'medium', 'classify',
            None, {}, 'success', {'labels': ['photograph'], 'scores': [0.87654321]}))
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            self.save(copy_images=False)
        self.save(folder='training', profile_id='composed', copy_images=False)
        report = self.save('native', copy_images=False, archive_all=True)
        archive = Path(report['destination']) / 'records.jsonl'
        self.assertEqual([r.to_dict() for r in read_records(archive)], [r.to_dict() for r in self.records])
        self.records[0].candidates[0].edit('Upstream text is now different.')
        report = self.save(folder='stale', profile_id='composed', copy_images=False)
        self.assertEqual(report['exported'], 2)
        self.assertIn('Stale', report['skipped'][0]['reason'])

    def test_protects_paths_and_rolls_back_a_failed_publish(self):
        symlink = self.root / 'link'
        symlink.symlink_to(self.root / 'a', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.save(folder='link/sub', copy_images=False)
        with self.assertRaisesRegex(ValueError, 'original image'):
            self.save(folder='a', copy_images=False)
        conflict = self.sources[1].with_name('photo.jpg.txt')
        conflict.symlink_to(self.sources[0])
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.save(beside=True, copy_images=False, overwrite=True)
        self.assertFalse(self.sources[0].with_name('photo.png.txt').exists())
        calls = []

        def fail_on_second(staged, target, overwrite):
            calls.append(target)
            if len(calls) == 2:
                raise OSError('disk failure')
            publish_file(staged, target, overwrite)

        with patch('captioning.export.publish_file', side_effect=fail_on_second):
            with self.assertRaisesRegex(OSError, 'disk failure'):
                self.save(copy_images=False)
        self.assertFalse((self.root / 'output').exists())

    def test_failed_output_flush_removes_partial_target(self):
        staged = self.root / 'staged.txt'
        staged.write_text('A caption.')
        target = self.root / 'target.txt'
        real_open = open

        class FailingClose:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                return self.stream

            def __exit__(self, *_):
                self.stream.close()
                raise OSError('disk full at close')

        def open_file(path, mode):
            stream = real_open(path, mode)
            return FailingClose(stream) if mode == 'xb' else stream

        with patch('captioning.records.open', side_effect=open_file, create=True):
            with self.assertRaisesRegex(OSError, 'disk full at close'):
                publish_file(staged, target)
        self.assertFalse(target.exists())

    def test_existing_files_are_restored_when_replacement_fails(self):
        self.save(copy_images=False)
        destination = self.root / 'output'
        before = {file.name: file.read_bytes() for file in destination.iterdir()}
        for record in self.records:
            record.candidates[0].edit('Edited after the first save.')
        calls = []

        def fail_second(staged, target, overwrite):
            calls.append(target)
            if len(calls) == 2:
                raise OSError('replace failure')
            publish_file(staged, target, overwrite)

        with patch('captioning.export.publish_file', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'replace failure'):
                self.save(copy_images=False, overwrite=True)
        self.assertEqual({file.name: file.read_bytes() for file in destination.iterdir()}, before)

    def test_legacy_explicit_replace_does_not_truncate_linked_source(self):
        node = load_extension().NODE_CLASS_MAPPINGS['CaptionExportTpl']()
        image = self.sources[0]
        original = image.read_bytes()
        sidecar = image.with_name(image.name + '.txt')
        os.link(image, sidecar)
        node.save_captions(['A caption.'], [image], keep_extension=True, overwrite_existing=True)
        self.assertEqual(image.read_bytes(), original)
        self.assertEqual(sidecar.read_text(), 'A caption.\n')

    def test_full_records_can_archive_previous_work_without_source_copies(self):
        original = [r.to_dict() for r in self.records]
        self.sources[0].unlink()
        report = self.save('native', copy_images=False, archive_all=True)
        records = Path(report['destination']) / 'records.jsonl'
        self.assertEqual([r.to_dict() for r in read_records(records)], original)
        with self.assertRaises(OSError):
            self.save('native', folder='copies', copy_images=True, archive_all=True)
        self.assertFalse((self.root / 'copies').exists())

    def test_sidecar_only_accepts_txt_and_no_copies(self):
        with self.assertRaisesRegex(ValueError, 'separate Output Folder'):
            self.save('csv', beside=True)
        with self.assertRaisesRegex(ValueError, 'Turn off Copy Images'):
            self.save(beside=True, copy_images=True)

    def test_output_toggle_saved_record_input_and_failed_captions(self):
        node = load_extension().NODE_CLASS_MAPPINGS['SaveCaptionsTpl']()
        destination = str(self.root / 'output')
        disabled = node.execute(destination=destination, file_policy='Create New Version')
        self.assertIn('off', disabled['result'][0])
        self.assertFalse((self.root / 'output').exists())
        self.records[0].candidates[0].status = 'failure'
        self.records[0].candidates[0].text = ''
        self.records[0].candidates[0].error = 'Pipeline returned no caption text'
        saved = self.root / 'saved.jsonl'
        write_records(self.records, saved)
        kwargs = dict(save_output=True, destination=destination, records_file=str(saved))
        result = node.execute(**kwargs)['result'][0]
        self.assertIn('Saved 2', result)
        self.assertIn('Skipped 1', result)
        self.assertIn('Pipeline returned no caption text', result)
        self.assertIn('Protected', node.execute(**kwargs)['result'][0])
