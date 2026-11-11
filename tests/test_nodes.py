from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from test_import import load_extension

extension = load_extension()
from tpl.nodes.model_loader_tpl import ModelLoaderTpl
from tpl.nodes.caption_generator_tpl import CaptionGeneratorTpl
from tpl.nodes.caption_export_tpl import CaptionExportTpl
from tpl.nodes.image_loader_tpl import ImageLoaderTpl
from tpl.nodes.batch_processor_tpl import BatchProcessorTpl


class NodesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = []
        for i in range(10):
            path = self.root / f"{i:02}.png"
            Image.new("RGB", (12 + i, 12), (i, 20, 30)).save(path)
            self.paths.append(str(path))

    def fake(self, task, callback, chat=False, florence=False):
        class FakePipeline:
            def __call__(self, *args, **kwargs):
                return callback(*args, **kwargs)
        pipe = FakePipeline()
        pipe.task = task
        pipe.processor = SimpleNamespace(chat_template="template" if chat else None)
        pipe.model = SimpleNamespace(config=SimpleNamespace(model_type="florence2" if florence else "test"))
        return pipe

    def test_ten_images_keep_order_across_uneven_batches(self):
        def run(image, **kwargs):
            self.assertEqual(kwargs["max_new_tokens"], 75)
            return [{"generated_text": str(image.width)}]
        pipe = self.fake("image-to-text", run)
        batches = BatchProcessorTpl().process_batches(self.paths, 3)[0]
        captions = CaptionGeneratorTpl().generate_captions(pipe, batches)[0]
        self.assertEqual(captions, ["12", "13", "14", "15", "16", "17", "18", "19", "20", "21"])

    def test_corrupt_image_fails_with_path_instead_of_shifting_results(self):
        Path(self.paths[4]).write_text("not an image")
        pipe = self.fake("image-to-text", lambda *a, **k: [{"generated_text": "caption"}])
        with self.assertRaisesRegex(RuntimeError, "04.png"):
            CaptionGeneratorTpl().generate_captions(pipe, [self.paths])

    def test_chat_extracts_assistant_only(self):
        def run(*args, **kwargs):
            messages = kwargs["text"]
            self.assertEqual(messages[0]["content"][0]["type"], "image")
            self.assertEqual(messages[0]["content"][1]["text"], "Describe style")
            return [{"generated_text": messages + [{"role": "assistant", "content": [{"type": "text", "text": "watercolour"}]}]}]
        for task in ("image-text-to-text", "any-to-any"):
            with self.subTest(task=task):
                captions = CaptionGeneratorTpl().generate_captions(
                    self.fake(task, run, chat=True), [[self.paths[0]]], question="Describe style"
                )[0]
                self.assertEqual(captions, ["watercolour"])

    def test_florence_uses_task_prompt_and_postprocessor(self):
        def run(**kwargs):
            self.assertEqual(kwargs["text"], "<CAPTION>")
            self.assertFalse(kwargs["skip_special_tokens"])
            return [{"generated_text": "<s>scene</s>"}]
        pipe = self.fake("image-text-to-text", run, florence=True)
        pipe.processor.post_process_generation = lambda text, task, image_size: {task: "scene"}
        self.assertEqual(CaptionGeneratorTpl().generate_captions(pipe, [[self.paths[0]]])[0], ["scene"])

    def test_visual_question_answering_normalizes_answer(self):
        def run(**kwargs):
            self.assertEqual(kwargs["question"], "What colour?")
            return [{"answer": "blue", "score": 0.9}]
        self.assertEqual(CaptionGeneratorTpl().generate_captions(
            self.fake("visual-question-answering", run), [[self.paths[0]]], question="What colour?"
        )[0], ["blue"])

    def test_modern_blip_does_not_receive_an_instruction_prompt(self):
        def run(**kwargs):
            self.assertEqual(kwargs["text"], "")
            return [{"generated_text": "a bird"}]
        pipe = self.fake("image-text-to-text", run)
        pipe.model.config.model_type = "blip"
        self.assertEqual(CaptionGeneratorTpl().generate_captions(pipe, [[self.paths[0]]])[0], ["a bird"])

    @patch("tpl.nodes.pipeline_utils.transformers_version", "5.22.2", create=True)
    def test_model_specific_inference_options_reach_chat_pipeline(self):
        def run(**kwargs):
            self.assertFalse(kwargs["enable_thinking"])
            self.assertEqual(kwargs["generate_kwargs"], {"do_sample": False})
            return [{"generated_text": "a bird"}]
        pipe = self.fake("image-text-to-text", run, chat=True)
        captions = CaptionGeneratorTpl().generate_captions(
            pipe, [[self.paths[0]]], pipeline_kwargs='{"enable_thinking":false,"generate_kwargs":{"do_sample":false}}'
        )[0]
        self.assertEqual(captions, ["a bird"])

    @patch("tpl.nodes.pipeline_utils.transformers_version", "4.57.3", create=True)
    def test_legacy_chat_rejects_silently_ignored_template_options(self):
        pipe = self.fake("image-text-to-text", lambda **kwargs: [{"generated_text": "wrong reasoning mode"}], chat=True)
        with self.assertRaisesRegex(RuntimeError, "Transformers 5"):
            CaptionGeneratorTpl().generate_captions(pipe, [[self.paths[0]]], pipeline_kwargs='{"enable_thinking":false}')

    def test_inference_options_cannot_replace_images_or_output_contract(self):
        pipe = self.fake("image-text-to-text", lambda **kwargs: [])
        for value in ('[]', '{"images":"different.png"}', '{"return_full_text":true}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CaptionGeneratorTpl().generate_captions(pipe, [self.paths], pipeline_kwargs=value)

    def test_bundled_example_has_valid_caption_and_export_widgets(self):
        from test_import import ROOT
        workflow = json.loads((ROOT / "example_workflows/Simple.json").read_text())
        nodes = {node["type"]: node for node in workflow["nodes"]}
        self.assertIsInstance(nodes["CaptionGeneratorTpl"]["widgets_values"][2], int)
        self.assertEqual(nodes["CaptionExportTpl"]["widgets_values"][0], "txt")

    def test_empty_and_ambiguous_outputs_fail_explicitly(self):
        for output in ([], [{"generated_text": ""}], [{"generated_text": "one"}, {"generated_text": "two"}], {"unknown": "x"}):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                CaptionGeneratorTpl().generate_captions(
                    self.fake("image-to-text", lambda *a, **k: output), [[self.paths[0]]]
                )

    def test_unsupported_caption_task_fails(self):
        with self.assertRaisesRegex(ValueError, "summarization"):
            CaptionGeneratorTpl().generate_captions(self.fake("summarization", lambda *a, **k: []), [self.paths])

    def test_export_rejects_count_mismatch_before_writing(self):
        with self.assertRaises(ValueError):
            CaptionExportTpl().save_captions(["one"], self.paths)
        self.assertEqual(list(self.root.glob("*.txt")), [])

    def test_export_overwrite_protection_and_utf8(self):
        node = CaptionExportTpl()
        result = node.save_captions(["café"] * 10, self.paths, keep_extension=False)
        self.assertEqual(result, ("Saved 10, Skipped 0",))
        self.assertEqual(node.save_captions(["new"] * 10, self.paths, keep_extension=False), ("Saved 0, Skipped 10",))
        self.assertEqual((self.root / "00.txt").read_text(encoding="utf-8"), "café\n")
        node.save_captions(["new"] * 10, self.paths, keep_extension=False, overwrite_existing=True)
        self.assertEqual((self.root / "00.txt").read_text(), "new\n")

    def test_export_rejects_image_extension_and_path_traversal(self):
        original = Path(self.paths[0]).read_bytes()
        for ext in ("png", "../txt", "txt/evil", "txt:stream"):
            with self.subTest(ext=ext), self.assertRaises(ValueError):
                CaptionExportTpl().save_captions(["oops"], self.paths[:1], file_extension=ext, keep_extension=False, overwrite_existing=True)
        self.assertEqual(Path(self.paths[0]).read_bytes(), original)

    def test_export_rejects_duplicate_destinations(self):
        other = self.root / "00.jpg"
        Image.new("RGB", (12, 12)).save(other)
        with self.assertRaises(ValueError):
            CaptionExportTpl().save_captions(["a", "b"], [self.paths[0], str(other)], keep_extension=False)
        self.assertFalse((self.root / "00.txt").exists())

    def test_invalid_directory_and_batch_size_fail(self):
        with self.assertRaises(ValueError):
            ImageLoaderTpl().load_images(str(self.root / "missing"))
        for size in (0, -1):
            with self.subTest(size=size), self.assertRaises(ValueError):
                BatchProcessorTpl().process_batches(self.paths, size)

    def test_loader_sorts_and_skips_both_caption_naming_styles(self):
        (self.root / "00.txt").write_text("caption")
        (self.root / "01.png.txt").write_text("caption")
        paths, directories = ImageLoaderTpl().load_images(str(self.root), True)
        self.assertEqual(paths, self.paths[2:])
        self.assertEqual(set(directories), set(paths))


class ModelTests(unittest.TestCase):
    @patch("tpl.nodes.model_loader_tpl.torch.cuda.is_available", return_value=True)
    @patch("tpl.nodes.model_loader_tpl.BitsAndBytesConfig")
    @patch("tpl.nodes.model_loader_tpl.pipeline")
    def test_quantized_load_preserves_trust_tokenizer_and_device_controls(self, factory, quantization, available):
        factory.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        pipe = ModelLoaderTpl().load_model("org/model", device_mode="cuda", use_bitsandbytes=True, quantization_type="8bit", trust_remote_code=True, use_fast_tokenizer=False)[0]
        self.assertEqual(pipe.device_map, {"": "cuda:0"})
        self.assertFalse(hasattr(pipe, "device"))
        self.assertTrue(pipe.trust_remote_code)
        self.assertFalse(pipe.use_fast)
        self.assertIs(pipe.model_kwargs["quantization_config"], quantization.return_value)
        self.assertTrue(quantization.call_args.kwargs["load_in_8bit"])
        self.assertFalse(quantization.call_args.kwargs["load_in_4bit"])

    @patch("tpl.nodes.model_loader_tpl.get_supported_tasks", return_value=["image-text-to-text"])
    @patch("tpl.nodes.model_loader_tpl.pipeline")
    def test_unavailable_task_fails_before_model_loading(self, factory, tasks):
        with self.assertRaisesRegex(ValueError, "any-to-any"):
            ModelLoaderTpl().load_model("org/model", "any-to-any", device_mode="cpu")
        factory.assert_not_called()

    @patch("tpl.nodes.model_loader_tpl.get_supported_tasks", return_value=["image-text-to-text"])
    @patch("tpl.nodes.model_loader_tpl.pipeline")
    def test_legacy_caption_task_maps_to_modern_task_when_removed(self, factory, tasks):
        factory.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        pipe = ModelLoaderTpl().load_model("Salesforce/blip-image-captioning-base", "image-to-text", device_mode="cpu")[0]
        self.assertEqual(pipe.task, "image-text-to-text")

    @patch("tpl.nodes.model_loader_tpl.pipeline")
    def test_cache_includes_task_device_and_loading_options(self, factory):
        factory.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        loader = ModelLoaderTpl()
        first = loader.load_model("org/model", device_mode="cpu")[0]
        self.assertIs(loader.load_model("org/model", device_mode="cpu")[0], first)
        changed = loader.load_model("org/model", task="text-generation", device_mode="cpu")[0]
        self.assertIsNot(changed, first)
        self.assertIsNot(loader.load_model("org/model", task="text-generation", device_mode="cpu", trust_remote_code=True)[0], changed)
        self.assertIsNot(loader.load_model("org/model", device_mode="cpu")[0], first)

    @patch("tpl.nodes.model_loader_tpl.pipeline", side_effect=OSError("unavailable model"))
    def test_loading_failure_is_not_none(self, factory):
        with self.assertRaisesRegex(RuntimeError, "org/missing"):
            ModelLoaderTpl().load_model("org/missing", device_mode="cpu")

    @patch("tpl.nodes.model_loader_tpl.torch.cuda.is_available", return_value=False)
    def test_explicit_cuda_does_not_silently_fall_back(self, available):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            ModelLoaderTpl().load_model("org/model", device_mode="cuda")

    @patch("tpl.nodes.model_loader_tpl.pipeline")
    def test_model_loading_options_reach_pipeline(self, factory):
        factory.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        pipe = ModelLoaderTpl().load_model("org/model", device_mode="cpu", dtype="float32", revision="abc", model_kwargs='{"attn_implementation":"eager"}')[0]
        self.assertEqual(pipe.revision, "abc")
        self.assertFalse(getattr(pipe, "local_files_only", pipe.model_kwargs.get("local_files_only")))
        self.assertEqual(pipe.model_kwargs["attn_implementation"], "eager")

    @patch("huggingface_hub.hf_hub_download", return_value="cached-snapshot/config.json")
    @patch("tpl.nodes.model_loader_tpl.pipeline")
    def test_offline_model_uses_resolved_local_snapshot(self, factory, snapshot):
        factory.side_effect = lambda **kwargs: SimpleNamespace(**kwargs)
        pipe = ModelLoaderTpl().load_model("org/model", device_mode="cpu", revision="abc", local_files_only=True)[0]
        self.assertEqual(pipe.model, "cached-snapshot")
        self.assertTrue(getattr(pipe, "local_files_only", pipe.model_kwargs.get("local_files_only")))
        snapshot.assert_called_once_with("org/model", "config.json", revision="abc", cache_dir=None, local_files_only=True)


if __name__ == "__main__":
    unittest.main()
