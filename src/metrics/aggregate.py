"""Per-run corpus scores followed by arithmetic means across runs."""
import csv
import random
from pathlib import Path
from .scoring import measure
from ..utils.config import Condition
from ..utils.io import read_json, read_text, write_json

NUMERIC = ["cer_before", "cer_after", "cer_delta", "edit_dist_before", "edit_dist_after", "N_gold",
           "len_asr", "len_out", "sh", "correct", "wf", "neutral", "total_ops", "destroyed", "Hb", "touched"]
METRICS = ["her", "ccdr", "cp", "cer_delta", "touch_rate"]


def micro(rows, metric):
    if metric == "her":
        num, den = sum(r["sh"] for r in rows), sum(r["total_ops"] for r in rows)
    elif metric == "cp":
        num, den = sum(r["correct"] for r in rows), sum(r["total_ops"] for r in rows)
    elif metric == "ccdr":
        num, den = sum(r["destroyed"] for r in rows), sum(r["Hb"] for r in rows)
    elif metric == "touch_rate":
        num, den = sum(r["touched"] for r in rows), sum(r["len_asr"] for r in rows)
    elif metric == "cer_delta":
        num = sum(r["edit_dist_after"] - r["edit_dist_before"] for r in rows)
        den = sum(r["N_gold"] for r in rows)
    else:
        raise ValueError(metric)
    return num / den if den else 0.0


def aggregate(rows):
    if not rows or sum(r["N_gold"] for r in rows) == 0:
        raise ValueError("Cannot evaluate an empty corpus or empty references")
    total = lambda key: sum(r[key] for r in rows)
    n_gold = total("N_gold")
    return {"n_files": len(rows), "ops": total("total_ops"), "correct": total("correct"),
            "sh": total("sh"), "wf": total("wf"), "neutral": total("neutral"),
            "net": total("correct") - total("sh"),
            "her": micro(rows, "her"), "cp": micro(rows, "cp"), "ccdr": micro(rows, "ccdr"),
            "touch_rate": micro(rows, "touch_rate"), "destroyed": total("destroyed"),
            "touched": total("touched"), "N_gold": n_gold,
            "cer_before": total("edit_dist_before") / n_gold,
            "cer_after": total("edit_dist_after") / n_gold,
            "cer_delta_micro": micro(rows, "cer_delta"),
            "n_improved": sum(r["cer_delta"] < -1e-12 for r in rows),
            "n_worse": sum(r["cer_delta"] > 1e-12 for r in rows),
            "n_same": sum(abs(r["cer_delta"]) <= 1e-12 for r in rows)}


def summarize(replicates):
    return {key: {"mean": sum(r[key] for r in replicates) / len(replicates),
                  "min": min(r[key] for r in replicates), "max": max(r[key] for r in replicates)}
            for key in replicates[0]}


def mean_rows(replicates):
    expected = [r["file"] for r in replicates[0]]
    if any([r["file"] for r in rows] != expected for rows in replicates):
        raise ValueError("Repetition file IDs or their order differ")
    return [{"file": name, **{key: sum(rows[i][key] for rows in replicates) / len(replicates)
                             for key in NUMERIC}} for i, name in enumerate(expected)]


def bootstrap(a, b, metric, samples=10000, seed=131071):
    if not a or [r["file"] for r in a] != [r["file"] for r in b]:
        raise ValueError("Bootstrap requires paired, identically ordered dialogue IDs")
    if samples < 40:
        raise ValueError("At least 40 bootstrap samples are needed for percentile intervals")
    rng, count = random.Random(seed), len(a)
    diffs = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        diffs.append(micro([a[i] for i in indices], metric) - micro([b[i] for i in indices], metric))
    diffs.sort()
    p = 2 * min(sum(d >= 0 for d in diffs), sum(d <= 0 for d in diffs)) / samples
    return {"observed_diff": micro(a, metric) - micro(b, metric),
            "ci95": [diffs[int(.025 * samples)], diffs[int(.975 * samples) - 1]],
            "p_two_sided": min(p, 1.0), "a": micro(a, metric), "b": micro(b, metric)}


def require_complete(root):
    status = read_json(root / "inference_status.json")
    if not status["complete"] or status["failures"]:
        raise ValueError("Refusing to evaluate incomplete inference")


def evaluate(args):
    root = Path(args.output)
    require_complete(root)
    manifest = read_json(root / "manifest.json")
    references = {name: read_text(Path(args.gold_dir) / name) for name in manifest["files"]}
    if any(not value.strip() for value in references.values()):
        raise ValueError("Empty reference text")
    result = {"conditions": {}}
    for spec in manifest["conditions"]:
        condition = Condition(**spec)
        repeats = manifest["settings"]["repeats"] if condition.stochastic else 1
        rep_rows = []
        for run in range(1, repeats + 1):
            rows = []
            for name in manifest["files"]:
                output = root / condition.name / f"run{run}" / name / "final.txt"
                metrics = measure(read_text(Path(manifest["paths"]["asr"]) / name), references[name], read_text(output))
                rows.append({"file": name, **{k: v for k, v in metrics.items() if k != "details"}})
            rep_rows.append(rows)
            write_json(root / condition.name / f"run{run}/per_file.json", rows)
        aggregates = [aggregate(rows) for rows in rep_rows]
        result["conditions"][condition.name] = {"repeats": repeats, "per_run": aggregates,
                                                "mean_per_file": mean_rows(rep_rows), "summary": summarize(aggregates)}
    write_json(root / "evaluation.json", result)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        columns = ["condition", "repeats", *next(iter(result["conditions"].values()))["summary"].keys()]
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, condition in result["conditions"].items():
            writer.writerow({"condition": name, "repeats": condition["repeats"],
                             **{k: v["mean"] for k, v in condition["summary"].items()}})
    print(f"Saved evaluation.json and summary.csv in {root}")


def run_bootstrap(args):
    root = Path(args.output)
    require_complete(root)
    evaluation = read_json(root / "evaluation.json")
    settings = read_json(root / "manifest.json")["settings"]
    names = evaluation["conditions"]
    pairs = [(a, b) for a, b in [("care", n) for n in names if n != "care"]
             + [("conservative", "identity"), ("dictionary", "conservative"), ("a_c", "a"), ("a_c", "a_b")]
             if a in names and b in names]
    result = {"samples": settings["bootstrap_samples"], "seed": settings["bootstrap_seed"], "pairs": {}}
    for a, b in pairs:
        result["pairs"][f"{a}-{b}"] = {metric: bootstrap(
            names[a]["mean_per_file"], names[b]["mean_per_file"], metric,
            samples=result["samples"], seed=result["seed"]) for metric in METRICS}
    write_json(root / "bootstrap.json", result)
    print(f"Saved {len(pairs)} paired comparisons; p=0 means p < 1 / samples")
