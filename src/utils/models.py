"""Explicit model download; never called during normal imports."""
from pathlib import Path

RELM_FILE_ID = "10vvkG_jzNK-CjIwlSvizhE1IOpnn9OqN"


def download_models(args):
    root = Path(args.model_dir)
    root.mkdir(parents=True, exist_ok=True)
    components = ["qwen", "bert", "relm"] if args.component == "all" else [args.component]
    for component in components:
        if component == "relm":
            import gdown
            output = root / "relm-m0.3.bin"
            if output.exists():
                continue
            if not gdown.download(id=RELM_FILE_ID, output=str(output)):
                raise RuntimeError("ReLM download failed; see models/README.md for manual download")
        else:
            local_name = "Qwen3.6-35B-A3B" if component == "qwen" else "bert-base-chinese"
            if args.source == "modelscope":
                from modelscope import snapshot_download
                model_id = "Qwen/Qwen3.6-35B-A3B" if component == "qwen" else "google-bert/bert-base-chinese"
                snapshot_download(model_id, revision=args.revision or "master", local_dir=str(root / local_name))
            else:
                from huggingface_hub import snapshot_download
                model_id = "Qwen/Qwen3.6-35B-A3B" if component == "qwen" else "google-bert/bert-base-chinese"
                snapshot_download(model_id, revision=args.revision or "main", local_dir=str(root / local_name))
