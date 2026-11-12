from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image
from captioning.classification import classify_text
from captioning.composition import compose_records
from captioning.export import export_records
from captioning.generation import generate_records
from captioning.merging import merge_records
from captioning.metadata import metadata_records
from captioning.profiles import CaptionProfile
from captioning.provenance import source_issue
from captioning.records import read_records, write_records
from test_import import load_extension


class Generator:
    task = 'text-generation'
    model = SimpleNamespace(name_or_path='fixture/generator', config=SimpleNamespace())
    tokenizer = SimpleNamespace(chat_template=None)

    def __call__(self, prompt, **kwargs):
        return [{'generated_text': 'A bird perched on a fence.'}]


class CompositionHardeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'source' / 'bird.png'
        self.image.parent.mkdir()
        Image.new('RGB', (12, 10), 'red').save(self.image)
        pipe = SimpleNamespace(task='image-text-to-text', model=SimpleNamespace(name_or_path='fixture/captioner', config=SimpleNamespace()))
        self.records = generate_records(pipe, [self.image], [CaptionProfile('facts', 'factual', 'describe')],
                                        infer=lambda *args: 'A bird on a fence.')
        self.image_bytes = self.image.read_bytes()

    def preview(self, records, profile):
        node = load_extension().NODE_CLASS_MAPPINGS['CaptionPreviewTpl']()
        return node.preview(caption_records=[r.to_dict() for r in records], profile_id=profile, show_images=False)['result'][0]

    def test_synthetic_text_is_clean_in_records_and_all_exports(self):
        records = compose_records(self.records, '{caption}', pipe=Generator(), profile_id='synthetic')
        candidate = records[0].candidates[-1]
        self.assertEqual(candidate.text, 'A bird perched on a fence.')
        self.assertEqual(candidate.generated_text, candidate.text)
        self.assertEqual(candidate.approval, 'unreviewed')
        self.assertEqual(candidate.generation_parameters['transformation']['verification'], 'unverified')
        self.assertIn('unverified', self.preview(records, 'synthetic').lower())
        for format in ('txt', 'imagefolder', 'native'):
            destination = self.root / format
            export_records(records, destination, format, profile_id='synthetic')
            if format == 'txt':
                text = next(destination.glob('*.txt')).read_text(encoding='utf-8').strip()
            elif format == 'imagefolder':
                text = json.loads((destination / 'metadata.jsonl').read_text(encoding='utf-8'))['text']
            else:
                text = read_records(destination / 'records.jsonl')[0].candidates[-1].text
            self.assertEqual(text, 'A bird perched on a fence.')
        self.assertEqual(self.image.read_bytes(), self.image_bytes)

    def test_legacy_status_prefix_is_removed_only_from_unedited_training_text(self):
        records = compose_records(self.records, '{caption}', pipe=Generator(), profile_id='synthetic')
        candidate = records[0].candidates[-1]
        candidate.text = 'Draft synthesis (unverified): ' + candidate.generated_text
        saved = self.root / 'source.jsonl'
        write_records(records, saved)
        original = saved.read_bytes()
        for format in ('txt', 'imagefolder'):
            destination = self.root / format
            export_records(read_records(saved), destination, format, profile_id='synthetic')
            text = (next(destination.glob('*.txt')).read_text(encoding='utf-8').strip() if format == 'txt'
                    else json.loads((destination / 'metadata.jsonl').read_text(encoding='utf-8'))['text'])
            self.assertEqual(text, candidate.generated_text)
        export_records(records, self.root / 'archive', 'native', profile_id='synthetic')
        self.assertEqual(read_records(self.root / 'archive/records.jsonl')[0].to_dict(), records[0].to_dict())
        candidate.edit('Draft synthesis (unverified): deliberately retained by editor')
        export_records(records, self.root / 'edited', 'txt', profile_id='synthetic')
        self.assertEqual(next((self.root / 'edited').glob('*.txt')).read_text(encoding='utf-8').strip(), candidate.text)
        self.assertEqual(saved.read_bytes(), original)

    def test_default_composition_excludes_previous_derivations(self):
        records = compose_records(self.records, '{caption}', profile_id='first')
        records = compose_records(records, '{caption}', pipe=Generator(), profile_id='synthetic')
        # Provenance still identifies derivations even if their caption type was changed.
        records[0].candidates[1].caption_type = 'factual'
        result = compose_records(records, '{caption}')
        self.assertEqual(result[0].candidates[-1].text, 'A bird on a fence.')
        self.assertEqual(result[0].candidates[-1].generation_parameters['transformation']['source_candidate_ids'],
                         [self.records[0].candidates[0].id])
        self.assertEqual(result[0].to_dict()['candidates'][:-1], records[0].to_dict()['candidates'])

    def test_explicit_profile_or_candidate_allows_chaining_and_empty_selection_does_not(self):
        records = compose_records(self.records, 'First: {caption}', profile_id='first')
        source = records[0].candidates[-1]
        source.review('approved')
        before = records[0].to_dict()
        for selection in ({'profile_ids': ['first']}, {'candidate_ids': [source.id]}):
            result = compose_records(records, 'Next: {caption}', selection=selection)
            self.assertEqual(result[0].candidates[-1].text, 'Next: First: A bird on a fence.')
            self.assertEqual(result[0].candidates[-1].generation_parameters['transformation']['source_candidate_ids'], [source.id])
        for selection in ({'candidate_ids': []}, {'profile_ids': []}):
            self.assertEqual(compose_records(records, '{caption}', selection=selection)[0].candidates[-1].status, 'failure')
        self.assertEqual(records[0].to_dict(), before)

    def test_upstream_edits_or_rejections_block_transitive_reuse_and_training_export(self):
        base = compose_records(self.records, '{caption}', profile_id='first')
        base = compose_records(base, '{caption}', selection={'profile_ids': ['first']}, profile_id='second')
        for change in ('edit', 'reject'):
            records = deepcopy(base)
            if change == 'edit':
                records[0].candidates[0].edit('A church.')
            else:
                records[0].candidates[0].review('rejected')
            before = records[0].to_dict()
            result = compose_records(records, '{caption}', selection={'profile_ids': ['second']})
            self.assertEqual(result[0].candidates[-1].status, 'failure')
            self.assertIn('source', result[0].candidates[-1].error.lower())
            self.assertIn('unsafe', self.preview(records, 'second').lower())
            with self.assertRaises(ValueError):
                export_records(records, self.root / change, 'txt', profile_id='second')
            export_records(records, self.root / (change + '-archive'), 'native', profile_id='second')
            self.assertEqual(records[0].to_dict(), before)

    def test_changed_analysis_or_missing_snapshot_blocks_chaining(self):
        records = merge_records(self.records, metadata_records([self.image]))
        records[0].analyses[0].values = {'ISO': {'value': '200', 'source': 'EXIF:34855'}}
        composed = compose_records(records, '{caption} {metadata}', profile_id='first')
        for change in ('analysis', 'snapshot'):
            altered = deepcopy(composed)
            if change == 'analysis':
                altered[0].analyses[0].values['ISO']['value'] = '400'
            else:
                del altered[0].candidates[-1].generation_parameters['transformation']['source_candidate_snapshots']
            result = compose_records(altered, '{caption}', selection={'profile_ids': ['first']})
            self.assertEqual(result[0].candidates[-1].status, 'failure')

    def test_text_classification_cannot_hide_a_stale_derived_caption(self):
        class Classifier:
            task = 'zero-shot-classification'
            model = SimpleNamespace(name_or_path='fixture/classifier', config=SimpleNamespace())
            def __call__(self, *args, **kwargs):
                return {'labels': ['bird', 'building'], 'scores': [.9, .1]}
        records = compose_records(self.records, '{caption}', profile_id='first')
        records = classify_text(Classifier(), records, ['bird', 'building'], 'subject', source_profile='first')
        records[0].candidates[0].edit('A church.')
        result = compose_records(records, '{classifications}')
        self.assertEqual(result[0].candidates[-1].status, 'failure')

    def test_shared_ancestors_do_not_cause_exponential_provenance_reads(self):
        records = self.records
        for _ in range(18):
            records = compose_records(records, '{caption}',
                                      selection={'candidate_ids': [c.id for c in records[0].candidates[-2:]]})
        reads = []
        class CountedParameters(dict):
            def get(self, key, default=None):
                reads.append(key)
                return super().get(key, default)
        for candidate in records[0].candidates:
            candidate.generation_parameters = CountedParameters(candidate.generation_parameters)
        self.assertIsNone(source_issue(records[0].candidates[-1], records[0].candidates, []))
        self.assertLessEqual(len(reads), 3 * len(records[0].candidates))

    def test_native_v2_profile_selection_archives_complete_records_and_reports_scope(self):
        records = merge_records(self.records, metadata_records([self.image]))
        records = compose_records(records, '{caption}', profile_id='composed')
        records[0].candidates[0].review('rejected')
        records[0].candidates[-1].review('approved')
        # Retain another profile's explicit failure as well as metadata and rejected sources.
        records = compose_records(records, '{caption}', selection={'profile_ids': ['missing']}, profile_id='failed')
        saved = self.root / 'saved.jsonl'
        write_records(records, saved)
        digest = hashlib.sha256(saved.read_bytes()).hexdigest()
        report = export_records(read_records(saved), self.root / 'archive', 'native', profile_id='composed', approved_only=True)
        self.assertEqual(read_records(self.root / 'archive/records.jsonl')[0].to_dict(), records[0].to_dict())
        self.assertEqual(report['semantics'], 'record_archive')
        self.assertEqual(report['archived_candidates'], 3)
        self.assertEqual(report['archived_analyses'], 1)
        self.assertIn('all candidates', report['scope'])
        node = load_extension().NODE_CLASS_MAPPINGS['DatasetExportTpl']()
        result = node.execute(destination=str(self.root / 'node-archive'), format='native', profile_id='composed', records_file=str(saved))
        self.assertIn('all candidates', result['ui']['text'][0])
        self.assertEqual(hashlib.sha256(saved.read_bytes()).hexdigest(), digest)
        self.assertEqual(self.image.read_bytes(), self.image_bytes)
