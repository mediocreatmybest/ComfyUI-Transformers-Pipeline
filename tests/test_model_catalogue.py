from types import SimpleNamespace
import unittest
from unittest.mock import patch

from captioning.catalogue import CATALOGUE, CUSTOM_MODEL, model_choices, resolve_task
from captioning.profiles import CUSTOM_PROFILES, STYLE_LABELS, profiles_for_pipeline
from test_import import load_extension


BLIP = 'Salesforce/blip-image-captioning-base'
FLORENCE = 'florence-community/Florence-2-base'
SMOL = 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'
QWEN = 'Qwen/Qwen3.5-4B'
CAPRL = 'internlm/CapRL-Qwen3VL-4B'
CLIP = 'openai/clip-vit-base-patch32'
DISTIL = 'MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33'
SMOLLM = 'HuggingFaceTB/SmolLM2-135M-Instruct'


class CatalogueTests(unittest.TestCase):
    def test_exact_ids_defaults_evidence_and_task_mappings(self):
        expected = {BLIP: 'image-to-text', FLORENCE: 'image-text-to-text', SMOL: 'image-text-to-text',
                    QWEN: 'image-text-to-text', CAPRL: 'image-text-to-text', CLIP: 'zero-shot-image-classification',
                    SMOLLM: 'text-generation', DISTIL: 'zero-shot-classification'}
        self.assertEqual({name: CATALOGUE[name]['task'] for name in expected}, expected)
        self.assertEqual(len(CATALOGUE), 30)
        self.assertEqual(len([m for m in CATALOGUE.values() if m.get('legacy')]), 18)
        self.assertEqual(model_choices()[0], CUSTOM_MODEL)
        self.assertEqual(len(model_choices()), 1 + sum(bool(model['task']) for model in CATALOGUE.values()))
        for model in CATALOGUE.values():
            with self.subTest(model=model['id']):
                self.assertTrue(model['runtime_note'])
                self.assertIn(model['evidence'], {'desktop_reported', 'metadata_checked', 'upstream', 'experimental', 'legacy_incompatible'})
                self.assertIs(type(model['remote_code']), bool)
                self.assertTrue(set(model.get('styles', {})) <= STYLE_LABELS.keys())

    def test_legacy_picker_returns_existing_ids_and_loads_new_labels(self):
        cls = load_extension().NODE_CLASS_MAPPINGS['PresetModelListTpl']
        self.assertEqual(cls.get_model_list()[0], BLIP)
        self.assertEqual(len(cls.get_model_list()), 18)
        self.assertEqual(cls().select_model(BLIP), (BLIP,))
        label = next(choice for choice in model_choices() if 'CLIP ViT-B/32' in choice)
        self.assertEqual(cls().select_model(label), (CLIP,))

    def test_task_registry_and_model_specific_constraints(self):
        four = {'image-to-text', 'image-text-to-text', 'text-generation', 'zero-shot-image-classification'}
        five = four - {'image-to-text'}
        self.assertEqual(resolve_task(BLIP, 'image-to-text', four, '4.57.3'), 'image-to-text')
        self.assertEqual(resolve_task(BLIP, 'image-to-text', five, '5.22.2'), 'image-text-to-text')
        with self.assertRaisesRegex(ValueError, r'SmolVLM2.*image-text-to-text'):
            resolve_task(SMOL, 'image-to-text', four, '4.57.3')
        with self.assertRaisesRegex(ValueError, 'Unsupported model/task'):
            resolve_task(CLIP, 'text-generation', four, '4.57.3')
        with self.assertRaisesRegex(ValueError, r'Wrong Transformers version.*Qwen3.5'):
            resolve_task(QWEN, 'image-text-to-text', four, '4.57.3')
        self.assertEqual(resolve_task(CAPRL, 'image-text-to-text', four, '4.57.3'), 'image-text-to-text')
        self.assertEqual(resolve_task('Salesforce/instructblip-vicuna-7b', 'image-to-text', four, '4.57.3'),
                         'image-to-text')
        with self.assertRaisesRegex(ValueError, 'Community'):
            resolve_task('microsoft/Florence-2-base', 'image-text-to-text', four, '4.57.3')
        with self.assertRaisesRegex(ValueError, r'Unsupported task.*any-to-any.*4.57.3'):
            resolve_task('/local/checkpoint', 'any-to-any', four, '4.57.3')
        self.assertEqual(resolve_task('/local/checkpoint', 'text-generation', four, '4.57.3'), 'text-generation')

    def test_loader_presets_preserve_advanced_parameters_and_custom_override(self):
        node = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']()
        pipe = SimpleNamespace(_tpl_loading_diagnostics={})
        with patch.object(node.model_cache, 'load', return_value=pipe) as load:
            choice = next(label for label in model_choices() if 'CLIP ViT-B/32' in label)
            node.load_model('org/ignored', 'text-generation', model_preset=choice,
                            dtype='float32', placement='cpu', local_files_only=True)
            policy = load.call_args.args[0]
            self.assertEqual((policy.model_id, policy.task), (CLIP, 'zero-shot-image-classification'))
            self.assertEqual((policy.dtype, policy.placement, policy.local_files_only), ('float32', 'cpu', True))
            self.assertFalse(policy.trust_remote_code)
            node.load_model('/some/checkpoint', 'text-generation', model_preset=CUSTOM_MODEL,
                            device_mode='cpu', trust_remote_code=True)
            policy = load.call_args.args[0]
            self.assertEqual((policy.model_id, policy.task, policy.trust_remote_code),
                             ('/some/checkpoint', 'text-generation', True))

    def test_unsupported_model_fails_before_configuration_or_weight_download(self):
        loader = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']()
        with patch('tpl.captioning.loading.inspect_model') as inspect:
            with self.assertRaisesRegex(ValueError, 'Unsupported model/task'):
                loader.load_model(CLIP, 'text-generation', placement='cpu')
            inspect.assert_not_called()

    def test_missing_smolvlm_processor_dependency_precedes_base_and_adapter_downloads(self):
        loader = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']()
        with (patch('tpl.captioning.catalogue.find_spec', return_value=None),
              patch('tpl.captioning.loading.inspect_model') as inspect,
              patch('tpl.captioning.adapters.resolve_adapter') as adapter):
            with self.assertRaisesRegex(ImportError, 'Missing dependency.*num2words'):
                loader.load_model(SMOL, 'image-text-to-text', placement='cpu',
                                  use_lora=True, lora_adapter='org/adapter')
            inspect.assert_not_called()
            adapter.assert_not_called()

    def test_unrecognized_configuration_reports_architecture_before_loading_weights(self):
        loader = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']()
        with (patch('tpl.captioning.loading.inspect_model', return_value=('org/model', None, {})),
              patch('tpl.captioning.loading.resolve_config', side_effect=KeyError('unknown_vlm')),
              patch('transformers.pipeline') as pipeline):
            with self.assertRaisesRegex(ValueError, 'Unsupported model architecture.*unknown_vlm'):
                loader.load_model('org/model', 'image-text-to-text', placement='cpu')
            pipeline.assert_not_called()

    def test_caption_styles_keep_legacy_profiles_and_use_model_specific_inputs(self):
        def pipe(model_id):
            return SimpleNamespace(model=SimpleNamespace(name_or_path=model_id))

        def profile(model, style, **kwargs):
            return profiles_for_pipeline(pipe(model), STYLE_LABELS.get(style, style), **kwargs)[0]

        blip = profile(BLIP, 'factual')
        self.assertEqual((blip.id, blip.prompt_mode, blip.effective_prompt), ('factual', 'raw', ''))
        self.assertEqual(profile(BLIP, 'factual', prompt_override='a photo of').effective_prompt, 'a photo of')
        detailed = profile(FLORENCE, 'detailed_description')
        self.assertEqual((detailed.id, detailed.prompt_mode, detailed.effective_prompt),
                         ('detailed_description', 'raw', '<DETAILED_CAPTION>'))
        with self.assertRaisesRegex(ValueError, 'Florence.*task token'):
            profile(FLORENCE, 'factual', prompt_override='Describe the picture')
        self.assertEqual(profile(SMOL, 'natural_language').prompt_mode, 'instruction')
        self.assertEqual(profile(SMOL, 'subject_tags').output_format, 'tags')
        with self.assertRaisesRegex(ValueError, 'BLIP.*Simple Description'):
            profile(BLIP, 'artistic_analysis')
        with self.assertRaisesRegex(ValueError, 'custom profiles'):
            profile('user/local', 'factual')
        legacy = profiles_for_pipeline(pipe(BLIP), CUSTOM_PROFILES, profile_ids='factual')
        self.assertEqual(legacy[0].prompt_mode, 'instruction')
        with self.assertRaisesRegex(ValueError, '[Cc]hoose.*custom profiles'):
            profiles_for_pipeline(pipe(BLIP), STYLE_LABELS['factual'], profiles_json='{}')
