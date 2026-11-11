import json
import unittest

from test_import import ROOT
from workflow_privacy import privacy_issues


class WorkflowPrivacyTests(unittest.TestCase):
    def test_all_serialized_example_fields_are_free_of_local_provenance(self):
        for file in (ROOT / 'example_workflows').rglob('*.json'):
            with self.subTest(file=file.relative_to(ROOT)):
                self.assertEqual(privacy_issues(json.loads(file.read_text(encoding='utf-8'))), [])

    def test_common_leaks_are_detected_in_nested_fields_and_embedded_json(self):
        leaks = [
            r'C:\Users\example-user\Pictures', 'D:/private/models', '/home/example-user/images',
            r'\\sample-host\share\models', 'person@example.org', 'render-box.local',
            'hostname: render-box', 'DESKTOP-SAMPLE', 'http://render-box:8188', 'http://192.168.1.2:8188',
            '10.0.0.4', '172.20.1.2', '127.0.0.1', 'http://[fd12::1]:8188',
            'RTX 4090', 'NVIDIA GPU', 'AMD Radeon', 'Intel Arc GPU',
            'Created by: Example User', 'Codex session', 'agent_id: example',
            'Desktop Transformers 4.57.3', 'Transformers 5.22.2',
            'isolated Transformers worker', 'Physically validated on a workstation',
        ]
        for leak in leaks:
            with self.subTest(leak=leak):
                for field in ['title', 'label', 'properties', 'widgets_values', 'instructions']:
                    self.assertTrue(privacy_issues({'nodes': [{field: [leak]}]}))
                self.assertTrue(privacy_issues({'widgets_values': [json.dumps({'note': leak})]}))
        for key in ['username', 'hostname', 'agent_id', 'runtime', 'node_versions', 'created_by', 'supplied_author']:
            with self.subTest(key=key):
                self.assertTrue(privacy_issues({'extra': {key: 'example'}}))

    def test_technical_settings_model_ids_and_relative_paths_are_allowed(self):
        public = {
            'model_name_or_path': 'Salesforce/blip-image-captioning-base',
            'alternatives': ['florence-community/Florence-2-base', 'Qwen/Qwen3.5-4B', 'nvidia/model'],
            'revision': '0123456789abcdef0123456789abcdef01234567',
            'task': 'image-text-to-text', 'device': 'cuda:0', 'placement': 'offload',
            'paths': ['./input/images', './output/dataset', 'images/portraits'],
            'documentation': 'https://huggingface.co/docs/transformers',
            'schema_version': 1, 'version': 0.4,
            'instructions': ['Requires CUDA, bitsandbytes and Accelerate.',
                             'Use a Transformers release supporting the model architecture.',
                             'Requires Transformers >=4.57 for native Florence support.',
                             'CPU offloading is allowed; actual placement depends on memory budgets.'],
        }
        self.assertEqual(privacy_issues(public), [])


if __name__ == '__main__':
    unittest.main()
