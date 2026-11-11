import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_extension():
    spec = importlib.util.spec_from_file_location(
        "tpl", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tpl"] = module
    spec.loader.exec_module(module)
    return module


class ImportTests(unittest.TestCase):
    def test_missing_optional_exif_dependency_does_not_hide_caption_nodes(self):
        with patch.dict(sys.modules, {"exifread": None}):
            extension = load_extension()
        original_ids = {"BatchProcessorTpl", "CaptionExportTpl", "CaptionGeneratorTpl", "Florence2NodeTpl",
                        "ImageLoaderTpl", "ModelLoaderTpl", "PresetModelListTpl", "TaskListTpl", "DebugNodeTpl",
                        "DebugModelNodeTpl", "ExifMetadataExtractorTpl"}
        self.assertTrue(original_ids.issubset(extension.NODE_CLASS_MAPPINGS))
        self.assertIn("ModelLoaderTpl", extension.NODE_CLASS_MAPPINGS)
        self.assertIn("Florence2NodeTpl", extension.NODE_CLASS_MAPPINGS)


if __name__ == "__main__":
    unittest.main()
