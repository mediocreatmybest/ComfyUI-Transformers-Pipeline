import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image, PngImagePlugin

from captioning.profiles import CaptionProfile, load_profiles
from captioning.records import CaptionRecord, read_records, write_records
from captioning.generation import generate_records
from captioning.export import export_records


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.paths = []
        for folder, width in (("a", 12), ("b", 24)):
            parent = self.source / folder
            parent.mkdir()
            path = parent / "étude.png"
            meta = PngImagePlugin.PngInfo()
            meta.add_text("Artist", "Supplied créateur")
            Image.new("RGB", (width, 10), "blue").save(path, pnginfo=meta)
            self.paths.append(path)
        self.pipe = SimpleNamespace(task="image-text-to-text", model=SimpleNamespace(
            name_or_path="org/test", config=SimpleNamespace(model_type="test", _commit_hash="abc")
        ))
        self.profiles = [CaptionProfile("facts", "factual", "Describe visible objects."),
                         CaptionProfile("style", "artistic_analysis", "Describe composition and edges.")]
        self.calls = []

    def infer(self, pipe, image, prompt, limit, options):
        self.calls.append((image.width, prompt, limit, options))
        return f"Blue étude, width {image.width}."

    def records(self, paths=None, profiles=None):
        return generate_records(self.pipe, paths or self.paths, profiles or self.profiles,
                                model_id="org/test", revision="abc", infer=self.infer)

    def test_profiles_are_editable_and_model_independent(self):
        profiles = load_profiles()
        self.assertEqual(set(profiles), {"factual", "natural_language", "subject_tags", "artistic_analysis", "lora_training"})
        file = self.root / "custom.json"
        file.write_text(json.dumps({"schema_version": 1, "profiles": [
            {"id": "custom", "caption_type": "style", "prompt": "Analyse edges.", "max_new_tokens": 45,
             "pipeline_kwargs": {"generate_kwargs": {"do_sample": False}}}
        ]}))
        custom = load_profiles(file)["custom"]
        self.assertEqual(custom.max_new_tokens, 45)
        self.assertEqual(custom.prompt, "Analyse edges.")

    def test_multiple_candidates_preserve_image_identity_and_provenance(self):
        records = self.records()
        self.assertEqual([r.image.width for r in records], [12, 24])
        self.assertNotEqual(records[0].image.id, records[1].image.id)
        self.assertEqual(len(records[0].image.sha256), 64)
        self.assertEqual([c.caption_type for c in records[0].candidates], ["factual", "artistic_analysis"])
        self.assertEqual(records[1].candidates[0].text, "Blue étude, width 24.")
        candidate = records[0].candidates[0]
        self.assertEqual(candidate.model.model_id, "org/test")
        self.assertEqual(candidate.model.revision, "abc")
        self.assertEqual(candidate.model.pipeline_task, "image-text-to-text")
        self.assertEqual(candidate.approval, "unreviewed")
        self.assertIn("Do not identify", candidate.prompt)
        self.assertEqual(candidate.generation_parameters["max_new_tokens"], 192)

    def test_author_metadata_stays_separate_from_model_output(self):
        records = self.records()
        self.assertEqual(records[0].image.author, "Supplied créateur")
        self.assertEqual(records[0].image.author_source, "source_metadata")
        self.assertNotIn("Supplied créateur", self.calls[0][1])
        supplied = generate_records(self.pipe, self.paths[:1], self.profiles[:1], infer=self.infer,
                                    authors={str(self.paths[0]): "Named by user"})
        self.assertEqual(supplied[0].image.author, "Named by user")
        self.assertEqual(supplied[0].image.author_source, "supplied")

    def test_invalid_image_keeps_position_and_failure_candidates(self):
        bad = self.source / "bad.png"
        bad.write_bytes(b"not an image")
        records = self.records([self.paths[0], bad, self.paths[1]])
        self.assertEqual(len(records), 3)
        self.assertEqual(Path(records[1].image.source_path), bad)
        self.assertTrue(all(c.status == "failure" and not c.text and c.error for c in records[1].candidates))
        self.assertEqual(records[2].candidates[0].text, "Blue étude, width 24.")
        self.assertEqual(len(self.calls), 4)

    def test_inference_failure_is_local_to_candidate(self):
        def infer(pipe, image, prompt, *args):
            if "edges" in prompt:
                raise RuntimeError("backend failure")
            return "visible blue rectangle"
        records = generate_records(self.pipe, self.paths, self.profiles, infer=infer)
        self.assertEqual([c.status for c in records[0].candidates], ["success", "failure"])
        self.assertEqual(records[1].candidates[0].text, "visible blue rectangle")

    def test_tags_are_structured_only_when_valid_json(self):
        profile = CaptionProfile("tags", "subject_tags", "Return tags.", output_format="tags")
        for output, status, tags in [('["blue", "étude"]', "success", ["blue", "étude"]),
                                     ('```json\n["blue", "étude"]\n```', "success", ["blue", "étude"]),
                                     ('blue, maybe', "failure", [])]:
            with self.subTest(output=output):
                records = generate_records(self.pipe, self.paths[:1], [profile], infer=lambda *args: output)
                self.assertEqual(records[0].candidates[0].status, status)
                self.assertEqual(records[0].candidates[0].tags, tags)

    def test_review_edit_resets_approval_and_preserves_generated_text(self):
        candidate = self.records()[0].candidates[0]
        candidate.review("approved")
        candidate.edit("Corrected café caption")
        self.assertEqual(candidate.approval, "unreviewed")
        self.assertTrue(candidate.edited)
        self.assertEqual(candidate.generated_text, "Blue étude, width 12.")
        candidate.review("approved")
        self.assertEqual(candidate.text, "Corrected café caption")

    def test_editing_tag_text_clears_stale_structured_tags(self):
        profile = CaptionProfile("tags", "subject_tags", "Return tags.", output_format="tags")
        records = generate_records(self.pipe, self.paths[:1], [profile], infer=lambda *args: '["cat", "red"]')
        candidate = records[0].candidates[0]
        candidate.edit("dog, blue")
        self.assertEqual(candidate.tags, [])
        candidate.edit("dog, blue", tags=["dog", "blue"])
        self.assertEqual(candidate.tags, ["dog", "blue"])

    def test_changed_export_membership_requires_fresh_destination(self):
        records = self.records()
        destination = self.root / "dataset"
        export_records(records, destination, "txt", profile_id="facts")
        records[0].candidates[0].review("rejected")
        with self.assertRaisesRegex(ValueError, "new destination"):
            export_records(records, destination, "txt", profile_id="facts", overwrite=True)
        result = export_records(records, self.root / "reviewed", "txt", profile_id="facts")
        self.assertEqual(result["exported"], 1)
        self.assertEqual(len(list((self.root / "reviewed").glob("*.txt"))), 1)

    def test_versioned_roundtrip_and_no_overwrite(self):
        records = self.records()
        file = self.root / "records.jsonl"
        write_records(records, file)
        self.assertEqual([r.to_dict() for r in read_records(file)], [r.to_dict() for r in records])
        with self.assertRaises(FileExistsError):
            write_records(records, file)
        value = records[0].to_dict()
        value["schema_version"] = 999
        with self.assertRaises(ValueError):
            CaptionRecord.from_dict(value)

    def test_disjoint_txt_export_cannot_mix_dataset_membership(self):
        records = self.records()
        destination = self.root / "dataset"
        export_records(records[:1], destination, "txt", profile_id="facts")
        with self.assertRaisesRegex(ValueError, "new destination"):
            export_records(records[1:], destination, "txt", profile_id="facts")
        self.assertEqual(len(list(destination.glob("*.txt"))), 1)

    def test_failure_cannot_masquerade_as_success_or_approval(self):
        value = self.records()[0].to_dict()
        value["candidates"][0]["status"] = "failure"
        with self.assertRaises(ValueError):
            CaptionRecord.from_dict(value)

    def test_success_requires_verified_image_metadata(self):
        value = self.records()[0].to_dict()
        value["image"]["sha256"] = None
        with self.assertRaises(ValueError):
            CaptionRecord.from_dict(value)

    def test_non_object_generation_options_are_rejected(self):
        with self.assertRaises(ValueError):
            generate_records(self.pipe, self.paths, self.profiles, infer=self.infer, pipeline_kwargs=[])

    def test_record_output_cannot_overwrite_original(self):
        records = self.records()
        before = self.paths[0].read_bytes()
        with self.assertRaises(ValueError):
            write_records(records, self.paths[0], overwrite=True)
        self.assertEqual(self.paths[0].read_bytes(), before)

    def test_txt_duplicates_unicode_and_source_unchanged(self):
        before = [p.read_bytes() for p in self.paths]
        records = self.records()
        destination = self.root / "txt"
        result = export_records(records, destination, "txt", profile_id="facts")
        self.assertEqual(result["exported"], 2)
        texts = sorted(p.read_text(encoding="utf-8").strip() for p in destination.glob("*.txt"))
        self.assertEqual(texts, ["Blue étude, width 12.", "Blue étude, width 24."])
        self.assertEqual(len(list(destination.glob("*.png"))), 2)
        self.assertEqual([p.read_bytes() for p in self.paths], before)
        self.assertFalse(list(self.source.rglob("*.txt")))

    def test_imagefolder_has_correct_relative_images_and_text(self):
        records = self.records()
        dest = self.root / "hf"
        export_records(records, dest, "imagefolder", profile_id="style")
        rows = [json.loads(line) for line in (dest / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue((dest / row["file_name"]).is_file())
            self.assertIn("Blue étude", row["text"])
            self.assertEqual(row["profile_id"], "style")

    def test_native_preserves_failed_records_and_rich_provenance(self):
        bad = self.source / "bad.png"
        bad.write_bytes(b"bad")
        records = self.records([self.paths[0], bad])
        dest = self.root / "native"
        export_records(records, dest, "native", profile_id="facts")
        restored = read_records(dest / "records.jsonl")
        self.assertEqual(restored[0].candidates[0].model.model_id, "org/test")
        self.assertEqual(restored[1].candidates[0].status, "failure")
        self.assertEqual(Path(restored[0].image.source_path), self.paths[0])

    def test_failed_and_rejected_candidates_are_not_training_rows(self):
        records = self.records()
        records[0].candidates[0].review("rejected")
        result = export_records(records, self.root / "hf", "imagefolder", profile_id="facts")
        self.assertEqual(result["exported"], 1)
        self.assertEqual(len(result["skipped"]), 1)

    def test_approved_only_export_requires_human_approval(self):
        records = self.records()
        records[0].candidates[0].review("approved")
        result = export_records(records, self.root / "txt", "txt", profile_id="facts", approved_only=True)
        self.assertEqual(result["exported"], 1)

    def test_ambiguous_candidates_require_explicit_choice(self):
        records = self.records()
        extra = self.records()[0].candidates[0]
        records[0].candidates.append(extra)
        with self.assertRaises(ValueError):
            export_records(records, self.root / "txt", "txt", profile_id="facts")
        result = export_records(records, self.root / "txt", "txt", profile_id="facts",
                                candidate_ids={records[0].image.id: extra.id})
        self.assertEqual(result["exported"], 2)

    def test_reexport_without_inference_and_conflict_preflight(self):
        records = self.records()
        calls = len(self.calls)
        dest = self.root / "txt"
        export_records(records, dest, "txt", profile_id="facts")
        before = {p.name: p.read_bytes() for p in dest.iterdir()}
        with self.assertRaises(FileExistsError):
            export_records(records, dest, "txt", profile_id="style")
        self.assertEqual({p.name: p.read_bytes() for p in dest.iterdir()}, before)
        export_records(records, dest, "txt", profile_id="style", overwrite=True)
        self.assertEqual(len(self.calls), calls)

    def test_changed_source_and_source_destination_rejected_before_writes(self):
        records = self.records()
        with self.assertRaises(ValueError):
            export_records(records, self.source, "txt", profile_id="facts")
        self.paths[1].write_bytes(b"changed")
        dest = self.root / "txt"
        with self.assertRaises(ValueError):
            export_records(records, dest, "txt", profile_id="facts")
        self.assertFalse(dest.exists())


if __name__ == "__main__":
    unittest.main()
