import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_import import load_extension


class LocalAdapterCacheTests(unittest.TestCase):
    def setUp(self):
        self.node = load_extension().NODE_CLASS_MAPPINGS['ModelLoaderTpl']
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)/'adapter'
        self.folder.mkdir()
        (self.folder/'adapter_config.json').write_text('{}')
        (self.folder/'adapter_model.safetensors').write_bytes(b'first')

    def fingerprint(self, **kwargs):
        return self.node.IS_CHANGED(use_lora=True, lora_adapter=str(self.folder), **kwargs)

    def test_unchanged_and_disabled(self):
        self.assertEqual(self.fingerprint(), self.fingerprint())
        with patch('pathlib.Path.open', side_effect=AssertionError('must not read')):
            self.assertEqual(self.node.IS_CHANGED(use_lora=False, lora_adapter=str(self.folder)),
                             self.node.IS_CHANGED())
            self.assertEqual(self.node.IS_CHANGED(use_lora=True, lora_adapter='org/adapter', adapter_revision='a'*40),
                             self.node.IS_CHANGED(use_lora=True, lora_adapter='org/adapter', adapter_revision='a'*40))

    def test_same_size_preserved_timestamp_in_place_and_atomic_replacement(self):
        file = self.folder/'adapter_model.safetensors'
        original = self.fingerprint()
        stamp = file.stat()
        file.write_bytes(b'other')
        os.utime(file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        changed = self.fingerprint()
        self.assertNotEqual(original, changed)
        temporary = self.folder/'replacement'
        temporary.write_bytes(b'first')
        os.utime(temporary, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        os.replace(temporary, file)
        self.assertEqual(original, self.fingerprint())

    def test_config_missing_files_and_restoration(self):
        original = self.fingerprint()
        config = self.folder/'adapter_config.json'
        config.write_text('{"r":8}')
        self.assertNotEqual(original, self.fingerprint())
        config.unlink()
        missing = self.fingerprint()
        self.assertEqual(missing, self.fingerprint())
        config.write_text('{}')
        self.assertEqual(original, self.fingerprint())
        (self.folder/'adapter_model.safetensors').unlink()
        self.assertNotEqual(original, self.fingerprint())
        config.unlink()
        self.folder.rmdir()
        self.assertEqual(self.fingerprint(), self.fingerprint())

    def test_revision_and_other_branch_are_independent(self):
        before = self.node.IS_CHANGED(use_lora=False)
        self.assertNotEqual(self.fingerprint(adapter_revision='first'), self.fingerprint(adapter_revision='second'))
        (self.folder/'adapter_model.safetensors').write_bytes(b'changed')
        self.assertEqual(before, self.node.IS_CHANGED(use_lora=False))
