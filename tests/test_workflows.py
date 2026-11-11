from copy import deepcopy
import json
from pathlib import PurePosixPath, PureWindowsPath
import unittest

from test_import import ROOT, load_extension
from workflow_helpers import editor_to_api


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classes = load_extension().NODE_CLASS_MAPPINGS
        cls.definitions = {name: {'input': node.INPUT_TYPES(), 'output': list(node.RETURN_TYPES),
                                  'output_node': getattr(node, 'OUTPUT_NODE', False)}
                           for name, node in cls.classes.items()}
        cls.folder = ROOT / 'example_workflows'

    def test_required_editor_workflows_exist(self):
        expected = {'Simple.json', 'Florence_Caption.json', 'Caption_Profiles.json',
                    'Dataset_Export.json', 'Quantised_Captioning.json'}
        self.assertTrue(expected <= {p.name for p in self.folder.glob('*.json')})

    def test_every_editor_graph_has_valid_registered_nodes_widgets_and_links(self):
        for file in self.folder.glob('*.json'):
            with self.subTest(workflow=file.name):
                workflow = json.loads(file.read_text(encoding='utf-8'))
                prompt = editor_to_api(workflow, self.definitions)
                self.assertTrue(prompt)
                self.assertGreaterEqual(workflow['last_node_id'], max(n['id'] for n in workflow['nodes']))
                self.assertGreaterEqual(workflow['last_link_id'], max([link[0] for link in workflow['links']], default=0))

    def test_api_exports_match_editor_graphs_including_legacy_defaults(self):
        self.assertEqual({p.name for p in self.folder.glob('*.json')},
                         {p.name for p in (self.folder / 'api').glob('*.json')})
        for file in self.folder.glob('*.json'):
            with self.subTest(workflow=file.name):
                api_file = self.folder / 'api' / file.name
                self.assertTrue(api_file.exists(), f'Missing API export: {file.name}')
                self.assertEqual(json.loads(api_file.read_text(encoding='utf-8')),
                    editor_to_api(json.loads(file.read_text(encoding='utf-8')), self.definitions))

    def test_paths_are_portable_and_overwrites_are_disabled(self):
        path_fields = {'directory_path', 'destination', 'records_path', 'records_file', 'profiles_file'}
        for file in self.folder.glob('*.json'):
            prompt = editor_to_api(json.loads(file.read_text(encoding='utf-8')), self.definitions)
            for node in prompt.values():
                for name, value in node['inputs'].items():
                    if name in path_fields and isinstance(value, str) and value:
                        self.assertFalse(PureWindowsPath(value).drive or PurePosixPath(value).is_absolute(), value)
                        self.assertNotIn('..', PurePosixPath(value).parts)
                    if name.startswith('overwrite'):
                        self.assertFalse(value)

    def test_loader_exposes_all_supported_policy_controls_with_help(self):
        inputs = self.classes['ModelLoaderTpl'].INPUT_TYPES()
        fields = {**inputs['required'], **inputs['optional']}
        expected = {'model_name_or_path', 'task', 'device_mode', 'use_bitsandbytes', 'quantization_type',
                    'trust_remote_code', 'use_fast_tokenizer', 'dtype', 'revision', 'local_files_only',
                    'model_kwargs', 'quantization', 'placement', 'gpu_index', 'gpu_memory', 'cpu_memory'}
        self.assertEqual(set(fields), expected)
        for name, spec in fields.items():
            self.assertTrue(spec[1].get('tooltip'), name)
        self.assertEqual(self.classes['ModelLoaderTpl'].RETURN_TYPES[0], 'MODEL')

    def test_validator_rejects_broken_connections_and_widget_types(self):
        workflow = json.loads((self.folder / 'Simple.json').read_text())
        broken = deepcopy(workflow)
        broken['links'][0][1] = 999
        with self.assertRaises(ValueError):
            editor_to_api(broken, self.definitions)
        broken = deepcopy(workflow)
        next(n for n in broken['nodes'] if n['type'] == 'CaptionGeneratorTpl')['widgets_values'][2] = '75'
        with self.assertRaises(ValueError):
            editor_to_api(broken, self.definitions)

    def test_dataset_export_uses_saved_profiles_without_inference(self):
        file = self.folder / 'Dataset_Export.json'
        self.assertTrue(file.exists())
        prompt = editor_to_api(json.loads(file.read_text()), self.definitions)
        self.assertEqual({n['class_type'] for n in prompt.values()}, {'DatasetExportTpl'})
        self.assertEqual({n['inputs']['format'] for n in prompt.values()}, {'txt', 'imagefolder'})
        profiles = editor_to_api(json.loads((self.folder / 'Caption_Profiles.json').read_text()), self.definitions)
        records = next(n['inputs'] for n in profiles.values() if n['class_type'] == 'CaptionRecordsTpl')
        from captioning.profiles import select_profiles
        selected = select_profiles(records['profile_ids'], data=json.loads(records['profiles_json']))
        self.assertGreaterEqual(len(selected), 2)
        for node in prompt.values():
            self.assertEqual(node['inputs']['records_file'], records['records_path'])
            self.assertIn(node['inputs']['profile_id'], {p.id for p in selected})

    def test_quantised_example_allows_offload_with_automatic_memory_budgets(self):
        file = self.folder / 'Quantised_Captioning.json'
        self.assertTrue(file.exists())
        prompt = editor_to_api(json.loads(file.read_text()), self.definitions)
        settings = next(n['inputs'] for n in prompt.values() if n['class_type'] == 'ModelLoaderTpl')
        from captioning.loading import legacy_policy
        policy = legacy_policy(settings['model_name_or_path'], quantization=settings['quantization'],
                               placement=settings['placement'], dtype=settings['dtype'],
                               gpu_memory=settings['gpu_memory'], cpu_memory=settings['cpu_memory'])
        self.assertEqual((policy.model_id, policy.mode, policy.placement, policy.gpu_memory, policy.cpu_memory),
                         ('Salesforce/blip-image-captioning-base', 'int8', 'offload', '', ''))


if __name__ == '__main__':
    unittest.main()
