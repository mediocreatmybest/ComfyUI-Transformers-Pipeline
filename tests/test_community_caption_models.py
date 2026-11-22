from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image
from transformers import __version__
from transformers.pipelines import check_task, get_supported_tasks

from captioning.catalogue import CATALOGUE, model_choices, preset_model, resolve_task
from captioning.pipeline import caption_image
from captioning.profiles import profiles_for_pipeline
from test_import import load_extension


LEGACY_FINE_TUNES = (
    'gokaygokay/Florence-2-Flux',
    'gokaygokay/Florence-2-Flux-Large',
    'MiaoshouAI/Florence-2-base-PromptGen-v2.0',
    'MiaoshouAI/Florence-2-large-PromptGen-v2.0',
)
PROMPTGEN = ('Disty0/Florence-2-base-PromptGen-v2.0', 'Disty0/Florence-2-large-PromptGen-v2.0')
CANDRA = 'candra/blip-image-captioning-finetuned'
CAPRL = 'internlm/CapRL-Qwen3VL-4B'
FLORENCE = 'florence-community/Florence-2-base'


class CommunityCaptionModelTests(unittest.TestCase):
    def test_legacy_fine_tunes_explain_incompatibility_before_downloading(self):
        presets = {preset_model(choice)['id'] for choice in model_choices()[1:]}
        node = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']()
        for model_id in LEGACY_FINE_TUNES:
            with self.subTest(model_id=model_id), patch('tpl.captioning.loading.inspect_model') as inspect:
                self.assertNotIn(model_id, presets)
                self.assertTrue(CATALOGUE[model_id]['remote_code'])
                for trust in (False, True):
                    with self.assertRaisesRegex(ValueError, 'native Florence.*image-text-to-text'):
                        node.load_model(model_id, 'image-text-to-text', placement='cpu', trust_remote_code=trust)
                inspect.assert_not_called()

    def test_preflight_only_blocks_the_audited_native_caption_route(self):
        model_id = LEGACY_FINE_TUNES[0]
        tasks = {'text-generation', 'image-text-to-text'}
        # Leave alternate tasks and unaudited explicit revisions to the runtime's usual checks.
        self.assertEqual(resolve_task(model_id, 'text-generation', tasks, __version__, trust_remote_code=True),
                         'text-generation')
        self.assertEqual(resolve_task(model_id, 'image-text-to-text', tasks, __version__,
                                      revision='another-explicit-revision'), 'image-text-to-text')
        with self.assertRaisesRegex(ValueError, 'native Florence'):
            resolve_task(model_id, 'image-text-to-text', tasks, __version__,
                         revision=CATALOGUE[model_id]['audited_revision'])

    def test_installed_pipeline_task_registry_maps_native_model_configs(self):
        from transformers import BlipConfig, Florence2Config, Qwen3VLConfig

        for model_id, config, architecture in (
            (CANDRA, BlipConfig, 'BlipForConditionalGeneration'),
            (PROMPTGEN[0], Florence2Config, 'Florence2ForConditionalGeneration'),
            (PROMPTGEN[1], Florence2Config, 'Florence2ForConditionalGeneration'),
            (CAPRL, Qwen3VLConfig, 'Qwen3VLForConditionalGeneration'),
        ):
            with self.subTest(model_id=model_id):
                model = CATALOGUE[model_id]
                self.assertFalse(model['remote_code'])
                task = resolve_task(model_id, model['task'], get_supported_tasks(), __version__)
                _, info, _ = check_task(task)
                self.assertTrue(any(auto_class._model_mapping[config].__name__ == architecture
                                    for auto_class in info['pt']
                                    if hasattr(auto_class, '_model_mapping') and config in auto_class._model_mapping))

    def test_promptgen_styles_use_checkpoint_specific_raw_tokens_and_text(self):
        for model_id in PROMPTGEN:
            pipe = SimpleNamespace(model=SimpleNamespace(name_or_path=model_id))
            for style, token in (('Detailed Description', '<DETAILED_CAPTION>'),
                                 ('Keyword Caption', '<GENERATE_TAGS>'),
                                 ('Artistic Details', '<ANALYZE>')):
                profile = profiles_for_pipeline(pipe, style)[0]
                self.assertEqual((profile.effective_prompt, profile.prompt_mode, profile.output_format),
                                 (token, 'raw', 'text'))
            mixed = profiles_for_pipeline(pipe, 'Detailed Description', prompt_override='<MIXED_CAPTION_PLUS>')[0]
            self.assertEqual(mixed.effective_prompt, '<MIXED_CAPTION_PLUS>')
        standard = SimpleNamespace(model=SimpleNamespace(name_or_path=FLORENCE))
        with self.assertRaisesRegex(ValueError, 'Florence.*task token'):
            profiles_for_pipeline(standard, 'Detailed Description', prompt_override='<GENERATE_TAGS>')
        with self.assertRaisesRegex(ValueError, 'Florence.*task token'):
            profiles_for_pipeline(pipe, 'Detailed Description', prompt_override='<DESCRIPTION>')

    def test_promptgen_keyword_token_reuses_florence_pipeline_and_postprocessor(self):
        pipe = Mock(return_value=[{'generated_text': '<s>sky, red clouds</s>'}])
        pipe.task = 'image-text-to-text'
        pipe.model.config.model_type = 'florence2'
        pipe.processor.post_process_generation.return_value = {'<GENERATE_TAGS>': 'sky, red clouds'}
        with Image.new('RGB', (12, 9)) as image:
            result = caption_image(pipe, image, '<GENERATE_TAGS>', 120)
        self.assertEqual(result, 'sky, red clouds')
        self.assertEqual(pipe.call_args.kwargs['text'], '<GENERATE_TAGS>')
        self.assertFalse(pipe.call_args.kwargs['skip_special_tokens'])
        pipe.processor.post_process_generation.assert_called_once_with(
            '<s>sky, red clouds</s>', task='<GENERATE_TAGS>', image_size=(12, 9))

    def test_candra_uses_image_only_keyword_text_not_json_tags(self):
        model = SimpleNamespace(name_or_path=CANDRA, config=SimpleNamespace(model_type='blip'))
        profile = profiles_for_pipeline(SimpleNamespace(model=model), 'Keyword Caption')[0]
        self.assertEqual((profile.id, profile.effective_prompt, profile.output_format), ('keyword_caption', '', 'text'))
        for task in ('image-to-text', 'image-text-to-text'):
            pipe = Mock(return_value=[{'generated_text': 'dog, beach'}])
            pipe.model, pipe.task = model, task
            pipe.processor = SimpleNamespace(chat_template=None)
            with Image.new('RGB', (8, 8)) as image:
                self.assertEqual(caption_image(pipe, image, profile.effective_prompt, 75), 'dog, beach')
            if task == 'image-to-text':
                self.assertNotIn('prompt', pipe.call_args.kwargs)
            else:
                self.assertEqual(pipe.call_args.kwargs['text'], '')

    def test_caprl_uses_images_in_the_chat_message(self):
        pipe = Mock(return_value=[{'generated_text': 'A caption.'}])
        pipe.task = 'image-text-to-text'
        pipe.model = SimpleNamespace(name_or_path=CAPRL, config=SimpleNamespace(model_type='qwen3_vl'))
        pipe._tpl_model_info = {'model_id': CAPRL}
        pipe.processor = SimpleNamespace(chat_template='fixture with image placeholders')
        profile = profiles_for_pipeline(pipe, 'Detailed Description')[0]
        with Image.new('RGB', (12, 16)) as image:
            self.assertEqual(caption_image(pipe, image, profile.effective_prompt, 128), 'A caption.')
        blocks = pipe.call_args.kwargs['text'][0]['content']
        self.assertEqual([block['type'] for block in blocks], ['image', 'text'])
        self.assertEqual(blocks[1]['text'], profile.effective_prompt)
        self.assertIn('torchvision', CATALOGUE[CAPRL]['dependencies'])
