"""Opt-in real backend regression using an externally supplied two-branch test graph."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from urllib.request import build_opener, ProxyHandler, Request


@unittest.skipUnless(os.environ.get('TPL_QUEUE_GRAPH') and os.environ.get('TPL_BACKEND_URL'),
                     'Requires an isolated running backend and two-branch queue fixture')
class LocalAdapterQueueTests(unittest.TestCase):
    def test_local_replacement_requeues_only_adapter_branch(self):
        # Fixture: loaders 2/5, Caption Records 3/6, preview outputs 4/7.
        graph = json.loads(Path(os.environ['TPL_QUEUE_GRAPH']).read_text())
        source = Path(graph['5']['inputs']['lora_adapter'])
        opener = build_opener(ProxyHandler({}))
        url = os.environ['TPL_BACKEND_URL'].rstrip('/')
        def request(path, data=None):
            req = Request(url+path, data=json.dumps(data).encode() if data is not None else None,
                          headers={'Content-Type':'application/json'})
            with opener.open(req, timeout=30) as response:
                return json.load(response)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            adapter = root/'adapter'
            adapter.mkdir()
            for filename in ('adapter_config.json', 'adapter_model.safetensors'):
                shutil.copyfile(source/filename, adapter/filename)
            graph['5']['inputs'].update(lora_adapter=str(adapter), adapter_revision='', use_lora=True)
            for node in ('3', '6'):
                graph[node]['inputs'].update(records_path=str(root/(node+'.jsonl')), overwrite_records=True)
            original = None
            for state in ('initial', 'unchanged', 'replaced', 'config', 'missing', 'restored'):
                weights = adapter/'adapter_model.safetensors'
                if state == 'replaced':
                    from safetensors.torch import load_file, save_file
                    tensors = load_file(str(weights))
                    key = next(iter(tensors))
                    tensors[key] = tensors[key] + 0.01
                    stamp = weights.stat()
                    replacement = root/'replacement.safetensors'
                    save_file(tensors, str(replacement))
                    os.utime(replacement, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
                    os.replace(replacement, weights)
                if state == 'config':
                    config = json.loads((adapter/'adapter_config.json').read_text())
                    config['lora_alpha'] *= 2
                    (adapter/'adapter_config.json').write_text(json.dumps(config))
                if state == 'missing':
                    weights.unlink()
                if state == 'restored':
                    for filename in ('adapter_config.json', 'adapter_model.safetensors'):
                        shutil.copyfile(source/filename, adapter/filename)
                pid = request('/prompt', {'prompt':graph})['prompt_id']
                deadline = time.monotonic()+600
                while time.monotonic() < deadline:
                    history = request('/history/'+pid)
                    if pid in history:
                        break
                    time.sleep(.5)
                self.assertIn(pid, history, 'Backend queue timed out')
                history = history[pid]
                self.assertEqual(history['status']['status_str'], 'error' if state == 'missing' else 'success')
                cached = set(n for kind, detail in history['status']['messages'] if kind == 'execution_cached'
                             for n in detail['nodes'])
                if state != 'initial':
                    self.assertIn('2', cached, 'Base-only branch should remain cached')
                if state == 'unchanged':
                    self.assertIn('5', cached)
                if state in ('replaced', 'config', 'missing'):
                    self.assertNotIn('5', cached)
                base = json.loads((root/'3.jsonl').read_text().splitlines()[0])['candidates'][0]
                self.assertEqual(base['status'], 'success')
                original = original or base['text']
                self.assertEqual(base['text'], original)
                if state != 'missing':
                    candidate = json.loads((root/'6.jsonl').read_text().splitlines()[0])['candidates'][0]
                    self.assertEqual(candidate['status'], 'success')
                    info = candidate['model']['runtime']['adapter']
                    self.assertTrue(info['active'] and info['weights_verified'])
                    self.assertEqual(info['sha256'], hashlib.sha256(weights.read_bytes()).hexdigest())
                    if state == 'config':
                        self.assertEqual(info['config']['lora_alpha'], config['lora_alpha'])
