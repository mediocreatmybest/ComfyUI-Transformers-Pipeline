from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from PIL import Image


class MergeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'fixture.jpg'
        image = Image.new('RGB', (16, 12), 'red')
        exif = Image.Exif()
        exif[34855], exif[33437], exif[33434] = 200, 5.6, (1, 500)
        image.save(self.path, exif=exif)

    def test_metadata_fixture_empty_and_missing_preserve_identities(self):
        from captioning.metadata import metadata_records
        empty = self.path.with_name('empty.png')
        Image.new('RGB', (16, 12)).save(empty)
        records = metadata_records([self.path, empty, self.path.with_name('missing.jpg')])
        self.assertEqual(records[0].analyses[0].values['ISO']['value'], '200')
        self.assertEqual(records[0].analyses[0].values['ISO']['source'], 'EXIF:34855')
        self.assertEqual(records[1].analyses[0].values, {})
        self.assertEqual(records[1].analyses[0].status, 'success')
        self.assertEqual(records[2].analyses[0].status, 'failure')
        self.assertEqual(len({r.image.id for r in records}), 3)

    def test_merge_deduplicates_without_mutating_and_rejects_conflicts(self):
        from captioning.metadata import metadata_records
        from captioning.merging import merge_records
        records = metadata_records([self.path])
        original = deepcopy(records[0].to_dict())
        merged = merge_records(records, [], list(reversed(records)))
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0].analyses), 1)
        self.assertEqual(records[0].to_dict(), original)
        conflict = deepcopy(records)
        conflict[0].analyses[0].values['ISO']['value'] = '400'
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            merge_records(records, conflict)
        conflict = deepcopy(records)
        conflict[0].image.sha256 = '0' * 64
        with self.assertRaisesRegex(ValueError, 'identity'):
            merge_records(records, conflict)

    def test_same_path_with_changed_contents_is_not_silently_split(self):
        from captioning.metadata import metadata_records
        from captioning.merging import merge_records
        first = metadata_records([self.path])
        Image.new('RGB', (16, 12), 'blue').save(self.path)
        second = metadata_records([self.path])
        with self.assertRaisesRegex(ValueError, 'identity'):
            merge_records(first, second)


if __name__ == '__main__':
    unittest.main()
