import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
from captioning.generation import generate_records
from captioning.profiles import CaptionProfile
from captioning.records import write_records
from test_import import load_extension, ROOT


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "original.png"
        Image.new("RGB", (8, 8), "red").save(self.image)
        self.pipe = SimpleNamespace(task="image-text-to-text", model=SimpleNamespace(
            name_or_path="org/model", config=SimpleNamespace(model_type="test", _commit_hash="abc")))

    def test_additive_nodes_generate_and_export_records(self):
        extension = load_extension()
        self.assertIn("CaptionRecordsTpl", extension.NODE_CLASS_MAPPINGS)
        self.assertIn("DatasetExportTpl", extension.NODE_CLASS_MAPPINGS)
        node = extension.NODE_CLASS_MAPPINGS["CaptionRecordsTpl"]()
        with patch("tpl.captioning.pipeline.caption_image", return_value="A red rectangle."):
            result = node.generate(self.pipe, [str(self.image)], "factual,artistic_analysis")
        self.assertEqual(len(result[0][0]["candidates"]), 2)
        exporter = extension.NODE_CLASS_MAPPINGS["DatasetExportTpl"]()
        exported = exporter.export(str(self.root / "dataset"), "imagefolder", "factual", caption_records=result[0])
        self.assertEqual(json.loads(exported[0])["exported"], 1)

    def test_standalone_export_and_review_need_no_pipeline(self):
        records = generate_records(self.pipe, [self.image], [CaptionProfile("facts", "factual", "describe")],
                                   infer=lambda *args: "A red rectangle.")
        checkpoint = self.root / "records.jsonl"
        write_records(records, checkpoint)
        command = [sys.executable, "-B", "-m", "captioning.worker"]
        reviewed = self.root / "reviewed.jsonl"
        result = subprocess.run(command + ["review", "--records", str(checkpoint), "--candidate-id",
                                records[0].candidates[0].id, "--text", "Une étude rouge.", "--approval", "approved",
                                "--output", str(reviewed)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(command + ["export", "--records", str(reviewed), "--destination", str(self.root / "dataset"),
                                "--format", "txt", "--profile", "facts", "--approved-only"],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["exported"], 1)
        self.assertEqual(next((self.root / "dataset").glob("*.txt")).read_text(encoding="utf-8"), "Une étude rouge.\n")

    def test_checkpoint_node_is_output_and_profile_file_invalidates_cache(self):
        node = load_extension().NODE_CLASS_MAPPINGS["CaptionRecordsTpl"]
        self.assertTrue(getattr(node, "OUTPUT_NODE", False))
        profile = self.root / "profiles.json"
        profile.write_text('{"prompt": "first"}')
        first = node.IS_CHANGED(profiles_file=str(profile))
        profile.write_text('{"prompt": "second"}')
        self.assertNotEqual(first, node.IS_CHANGED(profiles_file=str(profile)))
        self.assertEqual(node.IS_CHANGED(profiles_json="{}"), node.IS_CHANGED(profiles_json="{}"))


if __name__ == "__main__":
    unittest.main()
