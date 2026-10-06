"""File I/O and manifests shared by experiment entry points."""
import importlib.metadata
import json
import platform
from pathlib import Path
from .. import __version__

RESOURCES = Path(__file__).resolve().parents[1] / "resources"


def require_resources(*names):
    missing = sorted(name for name in names if not (RESOURCES / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Missing controlled-access CARE resources: " + ", ".join(missing)
            + ". Request access from the authors and restore the approved files under "
            + str(RESOURCES) + "; see README.md (Controlled Resources)."
        )


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_text(path):
    # Keep BOMs in inference inputs; only the scoring normalizer removes them.
    return Path(path).read_text(encoding="utf-8")


def write_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def environment():
    packages = {"care": __version__}
    for name in ["torch", "vllm", "transformers", "pypinyin", "rapidfuzz", "numpy"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}
