from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from PIL import Image
from captioning.classification import classify_images, classify_text
from captioning.composition import compose_records
from captioning.generation import generate_records
from captioning.metadata import metadata_records
from captioning.merging import merge_records
from captioning.profiles import CaptionProfile
from captioning.records import CaptionRecord, read_records, write_records
from captioning.export import export_records
from test_import import ROOT, load_extension


class ModularIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for i in range(10):
            path = self.root / f'{i:02}-café.png'; Image.new('RGB', (12+i, 10), 'red').save(path); self.paths.append(path)
        self.paths.insert(4, self.root / 'invalid.png'); self.paths[4].write_bytes(b'invalid')
        class Classifier:
            task = 'zero-shot-image-classification'
            model = SimpleNamespace(name_or_path='fixture/classifier', config=SimpleNamespace())
            def __call__(self, image, candidate_labels, **kw):
                return [{'label': label, 'score': 1/len(candidate_labels)} for label in candidate_labels]
        self.classifier = Classifier()
        self.pipe = SimpleNamespace(task='image-text-to-text', model=SimpleNamespace(name_or_path='fixture/captioner', config=SimpleNamespace()))

    def test_independent_reordered_branches_keep_ten_valid_associations_and_failures(self):
        captions = generate_records(self.pipe, self.paths, [CaptionProfile('facts','factual','describe'), CaptionProfile('style','artistic_analysis','style')], infer=lambda p,img,*a:f'Image width {img.width}')
        original = [r.to_dict() for r in captions]
        medium = classify_images(self.classifier, list(reversed(self.paths)), ['photo','painting'], 'medium')
        subject = classify_images(self.classifier, self.paths, ['bird','building'], 'subject')
        merged = merge_records(captions, medium, subject, metadata_records(self.paths))
        self.assertEqual(len(merged), 11)
        self.assertEqual([r.to_dict() for r in captions], original)
        for i, record in enumerate(merged):
            self.assertEqual(Path(record.image.source_path), self.paths[i])
            self.assertEqual({a.group_id for a in record.analyses}, {'medium','subject','exif'})
            self.assertEqual(len(record.candidates), 2)
        composed = compose_records(merged, '{caption} {classifications}', selection={'profile_ids':['facts']})
        self.assertEqual(sum(r.candidates[-1].status=='success' for r in composed), 10)
        saved = self.root / 'saved.jsonl'; write_records(composed, saved)
        restored = read_records(saved)
        for format in ('txt','native','imagefolder'):
            report=export_records(restored, self.root / format, format, profile_id='composed')
            self.assertEqual(report['exported'], 11 if format=='native' else 10)
        self.assertEqual(len(list((self.root/'txt').glob('*.txt'))),10)
        self.assertEqual(len(read_records(self.root/'native/records.jsonl')),11)
        with self.assertRaises(FileExistsError):
            export_records(restored,self.root/'txt','txt',profile_id='composed')

    def test_merge_does_not_lose_edits_or_approvals(self):
        originals=generate_records(self.pipe,self.paths[:1],[CaptionProfile('facts','factual','describe')],infer=lambda *a:'Original')
        branch=deepcopy(originals); branch[0].candidates[0].edit('Edited'); branch[0].candidates[0].review('approved')
        with self.assertRaisesRegex(ValueError,'Conflicting'):
            merge_records(originals,branch)
        combined=merge_records(branch,metadata_records(self.paths[:1]))
        result=compose_records(combined,'{caption}')
        self.assertEqual(result[0].candidates[0].generated_text,'Original')
        self.assertEqual(result[0].candidates[0].approval,'approved')
        self.assertEqual(result[0].candidates[-1].approval,'unreviewed')

    def test_header_only_metadata_can_merge_with_failed_pixel_decode(self):
        path=self.root/'truncated.jpg'; Image.new('RGB',(100,100),'red').save(path)
        path.write_bytes(path.read_bytes()[:-10])
        captions=generate_records(self.pipe,[path],[CaptionProfile('facts','factual','describe')],infer=lambda *a:'caption')
        self.assertEqual(captions[0].candidates[0].status,'failure')
        merged=merge_records(captions,metadata_records([path]))
        self.assertEqual(len(merged),1)
        self.assertEqual(merged[0].candidates[0].status,'failure')

    def test_derived_sources_are_precise_snapshotted_and_validated(self):
        records=generate_records(self.pipe,self.paths[:1],[CaptionProfile('facts','factual','describe'),CaptionProfile('style','artistic_analysis','style')],infer=lambda *a:'study')
        composed=compose_records(records,'{attributes}')
        transform=composed[0].candidates[-1].generation_parameters['transformation']
        self.assertEqual(transform['source_candidate_ids'],[records[0].candidates[1].id])
        self.assertEqual(transform['source_candidate_snapshots'][0]['text'],'study')
        broken=composed[0].to_dict()
        broken['candidates'][-1]['generation_parameters']['transformation']['source_candidate_ids']=['absent']
        with self.assertRaises(ValueError):
            CaptionRecord.from_dict(broken)

    def test_stale_or_rejected_text_classifications_are_not_used_as_new_evidence(self):
        records=generate_records(self.pipe,self.paths[:1],[CaptionProfile('facts','factual','describe')],infer=lambda *a:'A cat')
        class TextClassifier:
            task='zero-shot-classification'
            model=SimpleNamespace(name_or_path='fixture/text',config=SimpleNamespace())
            def __call__(self,*a,**kw):
                return {'labels':['animal','building'],'scores':[.9,.1]}
        classified=classify_text(TextClassifier(),records,['animal','building'],'subject')
        self.assertEqual(classified[0].analyses[0].settings['source_text'],'A cat')
        classified[0].candidates[0].edit('A church')
        result=compose_records(classified,'{caption} {classifications}')
        self.assertEqual(result[0].candidates[-1].status,'failure')
        self.assertIn('Stale',result[0].candidates[-1].error)
        classified[0].candidates[0].review('rejected')
        result=compose_records(classified,'{classifications}')
        self.assertEqual(result[0].candidates[-1].status,'failure')

    def test_combined_example_shares_classifier_and_composes_without_model(self):
        file=ROOT/'example_workflows/api/Analysis_Dataset.json'
        self.assertTrue(file.exists())
        prompt=json.loads(file.read_text())
        classifiers=[n for n in prompt.values() if n['class_type']=='ZeroShotImageClassificationTpl']
        self.assertEqual(len(classifiers),2)
        self.assertEqual(classifiers[0]['inputs']['model_pipeline'],classifiers[1]['inputs']['model_pipeline'])
        self.assertNotEqual(classifiers[0]['inputs']['group_id'],classifiers[1]['inputs']['group_id'])
        self.assertIn('CaptionComposeTpl',{n['class_type'] for n in prompt.values()})
        self.assertNotIn('CaptionSynthesisTpl',{n['class_type'] for n in prompt.values()})
        self.assertEqual(len(load_extension().NODE_CLASS_MAPPINGS),21)


if __name__=='__main__':
    unittest.main()
