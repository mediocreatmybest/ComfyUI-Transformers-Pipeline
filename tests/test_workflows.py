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
        expected = {name + '.json' for name in (
            'Analysis_Dataset', 'Analysis_Reexport', 'Caption_Composition', 'Caption_Filter',
            'Caption_My_Images', 'Caption_Profiles', 'Dataset_Export', 'Florence_Caption',
            'Metadata_Merge', 'Optional_LoRA', 'Quantised_Captioning', 'Simple', 'Text_Synthesis',
            'Zero_Shot_Classification', 'Zero_Shot_Text', 'Caption_Image_Details', 'Prepare_My_Dataset',
        )}
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
        path_fields = {'directory_path', 'destination', 'records_path', 'records_file', 'profiles_file', 'lora_adapter'}
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
                    'model_kwargs', 'quantization', 'placement', 'gpu_index', 'gpu_memory', 'cpu_memory', 'use_lora', 'lora_adapter', 'adapter_revision'}
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

    def test_starters_use_records_preview_and_opt_in_saving(self):
        for name in ('Caption_My_Images', 'Caption_Image_Details', 'Prepare_My_Dataset'):
            with self.subTest(workflow=name):
                graph = json.loads((self.folder / (name + '.json')).read_text())
                prompt = editor_to_api(graph, self.definitions)
                nodes = {node['class_type']: (key, node['inputs']) for key, node in prompt.items()}
                self.assertTrue(graph['groups'])
                self.assertFalse({'CaptionGeneratorTpl', 'CaptionExportTpl', 'DatasetExportTpl'} & nodes.keys())
                _, save = nodes['SaveCaptionsTpl']
                _, preview = nodes['CaptionPreviewTpl']
                self.assertFalse(save['save_output'])
                self.assertTrue(preview['show_images'])
                self.assertFalse(preview['records_file'])
                self.assertFalse(any(n['inputs'].get('records_path') for n in prompt.values()))
                if name == 'Prepare_My_Dataset':
                    self.assertNotIn('ModelLoaderTpl', nodes)
                    loader, settings = nodes['RecordMergeTpl']
                    self.assertTrue(settings['records_file'])
                    filter_id, filtering = nodes['CaptionFilterTpl']
                    self.assertEqual(filtering['caption_records'], [loader, 0])
                    self.assertEqual(preview['caption_records'], [loader, 0])
                    self.assertEqual(save['caption_records'], [filter_id, 0])
                    continue
                folder, _ = nodes['ImageLoaderTpl']
                loader, settings = nodes['ModelLoaderTpl']
                self.assertEqual(settings['model_name_or_path'], 'Salesforce/blip-image-captioning-base')
                generator, inputs = nodes['CaptionRecordsTpl']
                self.assertEqual((inputs['model_pipeline'], inputs['image_paths']), ([loader, 0], [folder, 0]))
                output = generator
                if name == 'Caption_Image_Details':
                    metadata, inputs = nodes['ImageMetadataTpl']
                    self.assertEqual(inputs['image_paths'], [folder, 0])
                    output, merge = nodes['RecordMergeTpl']
                    self.assertEqual((merge['records_a'], merge['records_b']), ([generator, 0], [metadata, 0]))
                self.assertEqual(preview['caption_records'], [output, 0])
                self.assertEqual(save['caption_records'], [output, 0])

    def test_dataset_export_reuses_a_saved_profile_and_includes_images(self):
        prompt = editor_to_api(json.loads((self.folder / 'Dataset_Export.json').read_text()), self.definitions)
        nodes = {node['class_type']: (key, node['inputs']) for key, node in prompt.items()}
        self.assertEqual(set(nodes), {'RecordMergeTpl', 'SaveCaptionsTpl', 'CaptionPreviewTpl'})
        loader, settings = nodes['RecordMergeTpl']
        _, output = nodes['SaveCaptionsTpl']
        self.assertEqual(output['caption_records'], [loader, 0])
        self.assertEqual(output['file_format'], 'Images + Captions (ImageFolder)')
        self.assertTrue(output['copy_images'])
        self.assertFalse(output['save_output'])
        profiles = editor_to_api(json.loads((self.folder / 'Caption_Profiles.json').read_text()), self.definitions)
        records = next(n['inputs'] for n in profiles.values() if n['class_type'] == 'CaptionRecordsTpl')
        archive = next(n['inputs'] for n in profiles.values() if n['class_type'] == 'SaveCaptionsTpl')
        from captioning.profiles import select_profiles
        selected = select_profiles(records['profile_ids'], data=json.loads(records['profiles_json']))
        self.assertGreaterEqual(len(selected), 2)
        self.assertTrue(settings['records_file'].endswith('/records.jsonl'))
        self.assertEqual(settings['records_file'].rsplit('/', 1)[0], archive['destination'])
        self.assertIn(output['profile_id'], {p.id for p in selected})

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

    def test_previews_cover_live_records_and_loaded_archives(self):
        for name, source in [('Caption_Profiles.json', 'caption_records'), ('Dataset_Export.json', 'caption_records')]:
            prompt = editor_to_api(json.loads((self.folder / name).read_text()), self.definitions)
            previews = [n['inputs'] for n in prompt.values() if n['class_type'] == 'CaptionPreviewTpl']
            self.assertEqual(len(previews), 1)
            self.assertTrue(previews[0][source])

    def test_example_node_rectangles_do_not_overlap_and_simple_has_widget_space(self):
        for file in self.folder.glob('*.json'):
            nodes = json.loads(file.read_text())['nodes']
            for i, a in enumerate(nodes):
                for b in nodes[i+1:]:
                    separated = (a['pos'][0] + a['size'][0] + 20 <= b['pos'][0] or
                                 b['pos'][0] + b['size'][0] + 20 <= a['pos'][0] or
                                 a['pos'][1] + a['size'][1] + 30 <= b['pos'][1] or
                                 b['pos'][1] + b['size'][1] + 30 <= a['pos'][1])
                    self.assertTrue(separated, (file.name, a['id'], b['id']))
            if file.name == 'Simple.json':
                for node in nodes:
                    minimum = {'ModelLoaderTpl': (430, 750), 'CaptionGeneratorTpl': (400, 380)}.get(node['type'])
                    if minimum:
                        self.assertGreaterEqual(node['size'][0], minimum[0])
                        self.assertGreaterEqual(node['size'][1], minimum[1])


if __name__ == '__main__':
    unittest.main()
