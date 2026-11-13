import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from captioning.loading import LoadingPolicy, PipelineCache


class AdapterTests(unittest.TestCase):
    def test_disabled_default_and_empty_enabled(self):
        self.assertFalse(LoadingPolicy('org/base').use_lora)
        with self.assertRaisesRegex(ValueError, 'LoRA adapter'):
            LoadingPolicy('org/base', use_lora=True)

    def test_local_identity_and_mismatch(self):
        from captioning.adapters import base_identity
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)/'models--org--base'/'snapshots'/('a'*40)
            snapshot.mkdir(parents=True)
            self.assertEqual(base_identity(str(snapshot)), ('org/base', 'a'*40))
            self.assertEqual(base_identity('org/base'), ('org/base', None))

    def test_config_checks(self):
        from captioning.adapters import validate_config
        good = {'peft_type':'LORA', 'base_model_name_or_path':'org/base', 'target_modules':['q_proj'], 'r':4}
        validate_config(good, 'org/base', 'a'*40)
        for changes, message in [({'peft_type':'PREFIX_TUNING'}, 'LoRA'),
                                  ({'base_model_name_or_path':'org/other'}, 'base'),
                                  ({'revision':'b'*40}, 'revision')]:
            with self.assertRaisesRegex(ValueError, message):
                validate_config({**good, **changes}, 'org/base', 'a'*40)

    def test_missing_local_config(self):
        from captioning.adapters import resolve_adapter
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'adapter_config'):
                resolve_adapter(LoadingPolicy('org/base', use_lora=True, lora_adapter=folder))

    def test_cache_failure_and_independence(self):
        from captioning import loading
        a, b = PipelineCache(), PipelineCache()
        with patch.object(loading, 'load_pipeline', side_effect=lambda policy: object()):
            pa = a.load(LoadingPolicy('org/base'))
            pb = b.load(LoadingPolicy('org/base'))
            self.assertIsNot(pa, pb)
            self.assertIs(pa, a.load(LoadingPolicy('org/base')))
            b.clear()
            self.assertIs(pa, a.pipeline)
        with patch.object(loading, 'load_pipeline', side_effect=RuntimeError('partial load')):
            with self.assertRaises(RuntimeError):
                b.load(LoadingPolicy('org/base'))
            self.assertIsNone(b.pipeline)

    def test_adapter_cache_uses_resolved_identity(self):
        from captioning import loading, adapters
        policy = LoadingPolicy('org/base', use_lora=True, lora_adapter='org/lora')
        cache = PipelineCache()
        with patch.object(adapters, 'resolve_adapter', return_value={'revision':'first'}) as resolve, patch.object(loading, 'load_pipeline', side_effect=lambda *a, **kw: object()) as load:
            first = cache.load(policy)
            self.assertIs(cache.load(policy), first)
            self.assertEqual(load.call_count, 1)
            resolve.return_value = {'revision':'second'}
            self.assertIsNot(cache.load(policy), first)
            self.assertEqual(load.call_count, 2)
            load.side_effect = RuntimeError('adapter failed')
            resolve.return_value = {'revision':'third'}
            with self.assertRaises(RuntimeError): cache.load(policy)
            self.assertIsNone(cache.pipeline)
            self.assertIsNone(cache.key)

    def test_local_base_alias_and_pinned_revision(self):
        from captioning.adapters import validate_config
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)/'models--org--base'/'snapshots'/('a'*40)
            snapshot.mkdir(parents=True)
            config = {'peft_type':'LORA','base_model_name_or_path':'org/base','revision':'a'*40,'target_modules':['q_proj'],'r':4}
            validate_config(config, str(snapshot), 'a'*40)
            with self.assertRaisesRegex(ValueError, 'revision'):
                validate_config(config, str(snapshot), 'b'*40)

    def test_ui_defaults_and_advanced_revision(self):
        from test_import import load_extension
        fields = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl'].INPUT_TYPES()['optional']
        self.assertFalse(fields['use_lora'][1]['default'])
        self.assertTrue(fields['adapter_revision'][1]['advanced'])
        self.assertEqual(list(fields)[-3:], ['use_lora','lora_adapter','adapter_revision'])

    def test_weights_verified_and_partial_activation_rejected(self):
        import torch
        from types import SimpleNamespace
        from safetensors.torch import save_file
        from captioning.adapters import activate_adapter
        weight = torch.ones(2, 2)
        with tempfile.TemporaryDirectory() as folder:
            save_file({'base_model.model.layer.lora_A.weight':weight}, str(Path(folder)/'adapter_model.safetensors'))
            layer = SimpleNamespace(lora_A=True, disable_adapters=False)
            model = SimpleNamespace(load_adapter=lambda *a, **k:None, active_adapters=lambda:['caption_lora'],
                                    named_parameters=lambda:[('layer.lora_A.caption_lora.weight',weight)], modules=lambda:[layer])
            resolved={'path':folder,'id':'org/lora','revision':'abc','sha256':'hash','config':{},'base_revision':'base'}
            with patch('peft.get_peft_model_state_dict', return_value={'layer.lora_A.weight':weight}):
                self.assertTrue(activate_adapter(SimpleNamespace(model=model),resolved)['weights_verified'])
                layer.disable_adapters=True
                with self.assertRaisesRegex(RuntimeError,'activation'):
                    activate_adapter(SimpleNamespace(model=model),resolved)
                layer.disable_adapters=False
            with patch('peft.get_peft_model_state_dict', return_value={'layer.lora_A.weight':torch.zeros(2,2)}):
                with self.assertRaisesRegex(RuntimeError,'partial'):
                    activate_adapter(SimpleNamespace(model=model),resolved)

    def test_missing_repository_or_revision_reports_config_failure(self):
        from captioning.adapters import resolve_adapter
        with patch('huggingface_hub.hf_hub_download', side_effect=OSError('not found')):
            with self.assertRaisesRegex(ValueError,'adapter_config.*revision'):
                resolve_adapter(LoadingPolicy('org/base',use_lora=True,lora_adapter='org/missing',adapter_revision='missing'))

    def test_unsupported_model_activation(self):
        from types import SimpleNamespace
        from captioning.adapters import activate_adapter
        with self.assertRaisesRegex(ValueError,'does not support'):
            activate_adapter(SimpleNamespace(model=object()),{})

    def test_caption_provenance_copies_adapter_without_changing_schema(self):
        from types import SimpleNamespace
        from captioning.classification import model_info
        from captioning.records import CaptionRecord
        pipe=SimpleNamespace(task='image-text-to-text',model=SimpleNamespace(name_or_path='org/base',config=SimpleNamespace()),
             _tpl_model_info={'model_id':'org/base','revision':'base','adapter':{'active':True,'id':'org/lora','revision':'adapter'}})
        info=model_info(pipe)
        self.assertEqual(info.runtime['adapter']['revision'],'adapter')
        info.runtime['adapter']['active']=False
        self.assertTrue(pipe._tpl_model_info['adapter']['active'])

    def test_missing_peft_and_missing_weights_are_actionable(self):
        from captioning.adapters import resolve_adapter
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'adapter_config.json'
            config.write_text(json.dumps({'peft_type':'LORA','base_model_name_or_path':'org/base','target_modules':['q_proj'],'r':4}))
            policy=LoadingPolicy('org/base',use_lora=True,lora_adapter=folder)
            with patch('captioning.loading.inspect_model',return_value=('org/base','a'*40,{})):
                with patch.dict('sys.modules',{'peft':None}):
                    with self.assertRaisesRegex(ValueError,'Installed PEFT'):
                        resolve_adapter(policy)
                with self.assertRaisesRegex(ValueError,'safetensors'):
                    resolve_adapter(policy)

    def test_bad_local_config_and_unverifiable_local_revision(self):
        from captioning.adapters import resolve_adapter
        with tempfile.TemporaryDirectory() as folder:
            config=Path(folder)/'adapter_config.json'
            config.write_text('[]')
            with self.assertRaisesRegex(ValueError,'JSON object'):
                resolve_adapter(LoadingPolicy('org/base',use_lora=True,lora_adapter=folder))
            config.write_text('{}')
            with self.assertRaisesRegex(ValueError,'revision'):
                resolve_adapter(LoadingPolicy('org/base',use_lora=True,lora_adapter=folder,adapter_revision='abc'))
