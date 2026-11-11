import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


class LoadingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('captioning.loading'), 'Shared loader is missing')
        from captioning import loading
        self.loading = loading

    def policy(self, **kwargs):
        return self.loading.LoadingPolicy('org/model', **kwargs)

    def assess(self, policy, config=None, dependencies=None):
        return self.loading.assess_capability(policy, config or {},
            dependencies if dependencies is not None else {'bitsandbytes': '0.50.1', 'accelerate': '1.12.0'}, True)

    def test_import_and_worker_help_do_not_import_inference_libraries(self):
        result = subprocess.run([sys.executable, '-B', '-c',
            'import sys; import captioning.loading; import captioning.worker; '
            'assert not ({"torch", "transformers", "accelerate", "bitsandbytes"} & sys.modules.keys())'],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_native_and_explicit_precision_are_distinct_from_quantisation(self):
        for dtype in ('auto', 'float16', 'bfloat16', 'float32'):
            policy = self.policy(dtype=dtype)
            self.assertEqual(policy.backend, 'native')
            self.assertEqual(self.assess(policy)['status'], 'potentially_supported')

    def test_missing_dependency_is_not_full_precision_fallback(self):
        result = self.assess(self.policy(backend='bitsandbytes', mode='int8'), dependencies={})
        self.assertEqual(result['status'], 'missing_optional_dependency')
        self.assertIn('bitsandbytes', result['reason'])

    def test_prequantised_checkpoint_is_preserved_or_conflict_rejected(self):
        config = {'quantization_config': {'quant_method': 'bitsandbytes', 'load_in_4bit': True}}
        self.assertEqual(self.assess(self.policy(), config)['status'], 'already_quantised')
        self.assertEqual(self.assess(self.policy(backend='bitsandbytes', mode='int8'), config)['status'], 'unsupported')

    def test_unverified_experimental_backends_are_not_advertised_as_supported(self):
        for backend in ('hqq', 'torchao'):
            result = self.assess(self.policy(backend=backend, mode='int8'), dependencies={backend: '1.0', 'accelerate': '1.12'})
            self.assertEqual(result['status'], 'unsupported')

    def test_invalid_controls_and_hidden_offload_options_fail(self):
        for values in ({'placement':'disk'}, {'gpu_memory':'all'}, {'cpu_memory':'-1GiB'},
                       {'backend':'bitsandbytes','mode':'fp4'}, {'dtype':'int8'},
                       {'model_kwargs':{'device_map':'auto'}}, {'model_kwargs':{'offload_folder':'x'}},
                       {'model_kwargs':{'load_in_8bit':True}}, {'model_kwargs':{'max_memory':{'cpu':'100GiB'}}}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.policy(**values)

    def test_cpu_budget_reserves_ram_and_rejects_oversized_request(self):
        budgets = self.loading.memory_budgets(self.policy(placement='cpu'), cpu_available=16*1024**3, gpu_free=None)
        self.assertEqual(budgets, {'cpu':8*1024**3})
        with self.assertRaisesRegex(ValueError, 'CPU'):
            self.loading.memory_budgets(self.policy(placement='cpu', cpu_memory='16GiB'), cpu_available=16*1024**3, gpu_free=None)

    def test_disk_plan_and_unrequested_cpu_placement_fail_before_loading(self):
        for mapping, placement in [({'a':0, 'b':'disk'}, 'offload'), ({'':'cpu'}, 'gpu'), ({'':'cpu'}, 'offload')]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                self.loading.validate_device_map(mapping, self.policy(placement=placement))

    def test_legacy_options_normalise_without_changing_saved_controls(self):
        policy = self.loading.legacy_policy('org/model', device_mode='cuda', use_bitsandbytes=True, quantization_type='4bit')
        self.assertEqual((policy.backend, policy.mode, policy.placement), ('bitsandbytes','nf4','gpu'))
        with self.assertRaises(ValueError):
            self.loading.legacy_policy('org/model', use_bitsandbytes=True, quantization='native')

    def test_cache_reuses_identical_policy_and_releases_old_reference(self):
        class Pipe:
            pass
        cache = self.loading.PipelineCache()
        with patch.object(self.loading, 'load_pipeline', side_effect=lambda policy: Pipe()):
            first = cache.load(self.policy())
            self.assertIs(first, cache.load(self.policy()))
            reference = weakref.ref(first)
            del first
            cache.load(self.policy(dtype='float16'))
            gc.collect()
            self.assertIsNone(reference())
            cache.clear()
            self.assertIsNone(cache.pipeline)

    def test_cache_key_includes_quantisation_placement_budgets_revision_and_options(self):
        cache = self.loading.PipelineCache()
        policies = [self.policy(), self.policy(backend='bitsandbytes', mode='int8'),
                    self.policy(backend='bitsandbytes', mode='nf4'), self.policy(placement='offload'),
                    self.policy(gpu_memory='2GiB'), self.policy(cpu_memory='3GiB'), self.policy(revision='abc'),
                    self.policy(model_kwargs={'attn_implementation':'eager'})]
        with patch.object(self.loading, 'load_pipeline', side_effect=lambda policy: SimpleNamespace(policy=policy)):
            results = [cache.load(policy) for policy in policies]
        self.assertEqual(len({id(result) for result in results}), len(policies))

    def test_full_precision_result_cannot_satisfy_quantised_request(self):
        import torch
        model = torch.nn.Linear(2, 2)
        with self.assertRaisesRegex(RuntimeError, 'quantisation'):
            self.loading.verify_quantisation(model, 'bitsandbytes', 'int8')

    def test_worker_and_node_build_equivalent_shared_policies(self):
        from captioning.worker import load_pipeline
        from test_import import load_extension
        node = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']()
        args = SimpleNamespace(model='org/model', task='image-to-text', device='cuda:0', dtype='float16',
            revision=None, local_files_only=True, trust_remote_code=False, model_kwargs='{}',
            quantization='int8', placement='offload', gpu_memory='2GiB', cpu_memory='4GiB')
        fake = SimpleNamespace(_tpl_loading_diagnostics={})
        with patch('captioning.loading.load_pipeline', return_value=fake) as worker_load:
            load_pipeline(args)
        with patch('tpl.captioning.loading.load_pipeline', return_value=fake) as node_load:
            node.load_model('org/model', dtype='float16', local_files_only=True, quantization='int8',
                            placement='offload', gpu_memory='2GiB', cpu_memory='4GiB')
        from dataclasses import asdict
        self.assertEqual(asdict(worker_load.call_args.args[0]), asdict(node_load.call_args.args[0]))

    def test_configuration_overrides_cannot_bypass_cpu_budget(self):
        from transformers import BertConfig
        with tempfile.TemporaryDirectory() as directory:
            BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=64, vocab_size=100).save_pretrained(directory)
            policy = self.loading.LoadingPolicy(directory, task='text-classification', placement='cpu',
                        model_kwargs={'num_hidden_layers':100}, local_files_only=True)
            with self.assertRaisesRegex(ValueError, 'budget'):
                self.loading.plan_placement(policy, directory, {}, {'cpu':128*1024})

    def test_subfolder_inspection_uses_the_actual_checkpoint_config(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'config.json').write_text('{}')
            (folder/'nested').mkdir()
            (folder/'nested'/'config.json').write_text('{"quantization_config":{"quant_method":"gptq"}}')
            _, _, config = self.loading.inspect_model(self.loading.LoadingPolicy(directory, model_kwargs={'subfolder':'nested'}))
            self.assertEqual(config['quantization_config']['quant_method'], 'gptq')

    def test_quantised_estimate_does_not_require_full_precision_gpu_capacity(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(1024, 1024), torch.nn.LayerNorm(1024))
        native = self.loading.estimate_weight_bytes(model, self.policy(dtype='float16'), {})
        nf4 = self.loading.estimate_weight_bytes(model, self.policy(backend='bitsandbytes', mode='nf4', dtype='float16'), {})
        self.assertLess(nf4, native / 2)
        self.assertGreater(nf4, 1024*1024 / 2)

    def test_prequantised_fp4_is_not_reported_as_nf4(self):
        self.assertEqual(self.loading.checkpoint_quantisation({'quant_method':'bitsandbytes', 'load_in_4bit':True,
                                                               'bnb_4bit_quant_type':'fp4'}), ('bitsandbytes','fp4'))

    def test_offload_planning_supports_installed_transformers_api(self):
        from transformers import BertConfig
        with tempfile.TemporaryDirectory() as directory:
            BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                       intermediate_size=64, vocab_size=100).save_pretrained(directory)
            policy = self.loading.LoadingPolicy(directory, task='text-classification', placement='offload', local_files_only=True)
            with patch('accelerate.infer_auto_device_map', return_value={'encoder':0, 'classifier':'cpu'}):
                mapping = self.loading.plan_placement(policy, directory, {}, {0:128*1024, 'cpu':1024*1024})
            self.assertEqual(set(mapping.values()), {0, 'cpu'})

    def test_actual_four_bit_format_is_verified(self):
        import torch
        layer_class = type('Linear4bit', (torch.nn.Linear,), {'__module__':'bitsandbytes.nn.modules'})
        layer = layer_class(2, 2)
        layer.weight.quant_type = 'fp4'
        with self.assertRaisesRegex(RuntimeError, 'type differs'):
            self.loading.verify_quantisation(layer, 'bitsandbytes', 'nf4')
        self.loading.verify_quantisation(layer, 'bitsandbytes', 'fp4')

    def test_diagnostics_redact_authentication_options(self):
        policy = self.policy(model_kwargs={'token':'test-secret', 'attn_implementation':'eager'})
        report = self.loading.policy_diagnostics(policy)
        self.assertNotIn('test-secret', json.dumps(report))
        self.assertEqual(report['model_kwargs']['attn_implementation'], 'eager')

    def test_versioned_config_is_used_for_capability_and_overrides_consumed(self):
        from transformers import BertConfig
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)
            config.configuration_files = ['config.4.0.0.json']
            config.save_pretrained(directory)
            selected = config.to_dict()
            selected['quantization_config'] = {'quant_method':'bitsandbytes','load_in_4bit':True,'bnb_4bit_quant_type':'nf4'}
            (root/'config.4.0.0.json').write_text(json.dumps(selected))
            policy = self.loading.LoadingPolicy(directory, backend='bitsandbytes', mode='int8',
                        model_kwargs={'num_hidden_layers':3, 'attn_implementation':'eager'}, local_files_only=True)
            effective, options = self.loading.resolve_config(policy, directory)
            self.assertEqual(effective.num_hidden_layers, 3)
            self.assertNotIn('num_hidden_layers', options)
            self.assertEqual(self.assess(policy, effective.to_dict())['status'], 'unsupported')

    def test_int8_layer_name_without_packed_weights_is_not_quantisation(self):
        import torch
        layer_class = type('Linear8bitLt', (torch.nn.Linear,), {'__module__':'bitsandbytes.nn.modules'})
        with self.assertRaisesRegex(RuntimeError, 'packed'):
            self.loading.verify_quantisation(layer_class(2, 2), 'bitsandbytes', 'int8')


if __name__ == '__main__':
    unittest.main()
