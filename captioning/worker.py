"""Run with python -m captioning.worker generate|export|review --help."""

import argparse
import json
from pathlib import Path
import sys

from .export import export_records
from .profiles import select_profiles
from .records import read_records, write_records


def load_pipeline(args):
    # Export and review deliberately do not import Torch or Transformers.
    import huggingface_hub
    import torch
    import transformers
    from transformers.pipelines import get_supported_tasks

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    task = args.task
    tasks = get_supported_tasks()
    if task == "image-to-text" and task not in tasks:
        task = "image-text-to-text"
    if task not in tasks:
        raise ValueError(f"Task {task} is unavailable in Transformers {transformers.__version__}")
    model = args.model
    options = json.loads(args.model_kwargs)
    if not isinstance(options, dict):
        raise ValueError("model-kwargs must be a JSON object")
    if set(options) & {"device", "device_map", "revision", "trust_remote_code", "dtype", "torch_dtype", "local_files_only"}:
        raise ValueError("Use dedicated loader arguments for device, revision, dtype, trust and offline mode")
    if args.local_files_only and not Path(model).is_dir():
        model = str(Path(huggingface_hub.hf_hub_download(
            model, "config.json", revision=args.revision, cache_dir=options.get("cache_dir"), local_files_only=True
        )).parent)
    kwargs = dict(task=task, model=model, device=args.device, dtype=args.dtype, model_kwargs=options,
                  trust_remote_code=args.trust_remote_code)
    if args.revision:
        kwargs["revision"] = args.revision
    if int(transformers.__version__.split(".")[0]) >= 5:
        kwargs["local_files_only"] = args.local_files_only
    else:
        options["local_files_only"] = args.local_files_only
    pipe = transformers.pipeline(**kwargs)
    revision = getattr(pipe.model.config, "_commit_hash", None)
    if revision is None and Path(model).parent.name == "snapshots":
        revision = Path(model).name
    pipe._tpl_model_info = {"model_id": args.model, "revision": revision or args.revision}
    return pipe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate versioned records; invalid images remain as failures")
    generate.add_argument("--images", required=True, type=Path)
    generate.add_argument("--model", required=True)
    generate.add_argument("--task", default="image-text-to-text")
    generate.add_argument("--profiles", default="factual,artistic_analysis")
    generate.add_argument("--profiles-file", type=Path)
    generate.add_argument("--pipeline-kwargs", default="{}")
    generate.add_argument("--model-kwargs", default="{}")
    generate.add_argument("--authors", type=Path, help="JSON mapping of source paths to supplied authors")
    generate.add_argument("--device", default="cpu")
    generate.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    generate.add_argument("--revision")
    generate.add_argument("--local-files-only", action="store_true")
    generate.add_argument("--trust-remote-code", action="store_true")
    generate.add_argument("--output", required=True, type=Path)
    generate.add_argument("--overwrite", action="store_true")
    export = commands.add_parser("export", help="Export saved records without running inference")
    export.add_argument("--records", required=True, type=Path)
    export.add_argument("--destination", type=Path, default=Path("caption_exports"))
    export.add_argument("--format", choices=["txt", "imagefolder", "native"], required=True)
    export.add_argument("--profile", required=True)
    export.add_argument("--caption-type")
    export.add_argument("--candidate-ids", default="{}", help="JSON image-id to candidate-id selection")
    export.add_argument("--approved-only", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    review = commands.add_parser("review", help="Save a reviewed/edited copy of the records")
    review.add_argument("--records", required=True, type=Path)
    review.add_argument("--candidate-id", required=True)
    review.add_argument("--text")
    review.add_argument("--tags", help="Replacement tags as a JSON array; also forms the caption if --text is omitted")
    review.add_argument("--approval", choices=["unreviewed", "approved", "rejected"], default="unreviewed")
    review.add_argument("--output", required=True, type=Path)
    review.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_records(read_records(args.records), args.destination, args.format,
                                    profile_id=args.profile, caption_type=args.caption_type,
                                    candidate_ids=json.loads(args.candidate_ids), overwrite=args.overwrite,
                                    approved_only=args.approved_only)
        elif args.command == "review":
            records = read_records(args.records)
            matches = [c for r in records for c in r.candidates if c.id == args.candidate_id]
            if len(matches) != 1:
                raise ValueError("Candidate id must match exactly one candidate")
            tags = json.loads(args.tags) if args.tags is not None else None
            if args.text is not None or tags is not None:
                text = args.text if args.text is not None else ", ".join(tags)
                matches[0].edit(text, tags=tags)
            matches[0].review(args.approval)
            write_records(records, args.output, args.overwrite)
            result = {"output": str(args.output), "candidate_id": args.candidate_id, "approval": args.approval}
        else:
            from .generation import generate_records
            if args.output.exists() and not args.overwrite:
                raise FileExistsError(f"Record output already exists: {args.output}")
            if not args.images.is_dir():
                raise ValueError("Images must be an existing directory")
            paths = sorted(p for p in args.images.rglob("*") if p.is_file() and
                           p.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"})
            if not paths:
                raise ValueError("No image files found")
            profiles = select_profiles(args.profiles, args.profiles_file)
            options = json.loads(args.pipeline_kwargs)
            authors = json.loads(args.authors.read_text(encoding="utf-8")) if args.authors else None
            pipe = load_pipeline(args)
            records = generate_records(pipe, paths, profiles, pipeline_kwargs=options, authors=authors)
            write_records(records, args.output, args.overwrite)
            failures = sum(c.status == "failure" for r in records for c in r.candidates)
            result = {"output": str(args.output), "images": len(records),
                      "candidates": sum(len(r.candidates) for r in records), "failures": failures}
            print(json.dumps(result, ensure_ascii=True))
            return 2 if failures else 0
        print(json.dumps(result, ensure_ascii=True))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
