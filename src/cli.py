"""Command line entry points for inference and independent analysis."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from .utils.config import CONDITIONS, DEFAULT_CONFIG


def add_prepare_options(parser):
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", choices=["main", "ablation", "sensitivity", "all"], default="main")
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS))
    parser.add_argument("--asr-dir", required=True, type=Path)
    parser.add_argument("--qwen-model", type=Path, default=Path("models/Qwen3.6-35B-A3B"))
    parser.add_argument("--bert-model", type=Path, default=Path("models/bert-base-chinese"))
    parser.add_argument("--relm-weights", type=Path, default=Path("models/relm-m0.3.bin"))
    parser.add_argument("--csc-device", default="cuda")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--files", type=int, nargs="+")


def run_all(args):
    # Separate processes release ReLM's CUDA memory before Qwen is loaded.
    # The child inherits an explicit package path so it does not depend on the caller's cwd.
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    command = [sys.executable, "-m", "care", "prepare"]
    for name in ["config", "suite", "asr_dir", "qwen_model", "bert_model", "relm_weights", "csc_device", "output", "repeats", "conditions", "files"]:
        value = getattr(args, name)
        if value is not None:
            command += ["--" + name.replace("_", "-")]
            command += [str(item) for item in value] if isinstance(value, list) else [str(value)]
    subprocess.run(command, check=True, env=environment)
    subprocess.run([sys.executable, "-m", "care", "infer", "--output", str(args.output)],
                   check=True, env=environment)
    if args.gold_dir:
        subprocess.run([sys.executable, "-m", "care", "evaluate", "--output", str(args.output),
                        "--gold-dir", str(args.gold_dir)], check=True, env=environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description="CARE local inference and paper experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "prepare", "infer", "evaluate", "bootstrap", "case-study", "plot"]:
        command = sub.add_parser(name)
        command.add_argument("--output", required=True, type=Path)
        if name in {"run", "prepare"}:
            add_prepare_options(command)
        if name in {"run", "evaluate", "case-study"}:
            command.add_argument("--gold-dir", type=Path, required=name != "run")
        if name in {"case-study", "plot"}:
            command.add_argument("--condition", default="care", choices=sorted(CONDITIONS))
        if name == "case-study":
            command.add_argument("--nogold-policy", choices=["paper", "separate"], default="separate")
        if name == "plot":
            command.add_argument("--kind", choices=["threshold", "cases"], required=True)
    command = sub.add_parser("download-models")
    command.add_argument("--component", choices=["qwen", "bert", "relm", "all"], default="all")
    command.add_argument("--model-dir", type=Path, default=Path("models"))
    command.add_argument("--source", choices=["modelscope", "huggingface"], default="modelscope")
    command.add_argument("--revision")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_all(args)
    if args.command == "prepare":
        from .pipeline import prepare
        return prepare(args)
    if args.command == "infer":
        from .vllm import infer
        return infer(args)
    if args.command in {"evaluate", "bootstrap"}:
        from .metrics.aggregate import evaluate, run_bootstrap
        return (evaluate if args.command == "evaluate" else run_bootstrap)(args)
    if args.command == "case-study":
        from .analysis.cases import analyze
        return analyze(args)
    if args.command == "plot":
        from .analysis.plots import plot
        return plot(args)
    from .utils.models import download_models
    return download_models(args)
