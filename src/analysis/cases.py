"""Proposal-level attribution, separate from output edit-block metrics."""
import csv
from collections import Counter
from pathlib import Path
from ..utils.io import RESOURCES, read_json, read_text, require_resources, write_json
from ..metrics.aggregate import require_complete
from ..metrics.alignment import GoldAligner, strip_punct, _s1_offset_table, _map_s2

CATEGORY_MAP = {**dict.fromkeys(["drug", "pharm", "dose"], "drug"),
                **dict.fromkeys(["param", "unit", "contact", "stimulation", "frequency", "model", "device", "target"], "parameter"),
                **dict.fromkeys(["symptom", "disease", "body", "exam", "action", "rehab"], "symptom")}
CATEGORIES = ["drug", "symptom", "parameter", "other"]


def classify(s1, s2, review, gold, entries, filename):
    aligner = GoldAligner(s1["source_text"], gold)
    offsets = _s1_offset_table(s1["corrections"])
    accepted = {d["diff_id"] for d in review["applied"]}
    decisions = {d["diff_id"]: d for d in review["decisions"]}
    rows = []
    for source, candidates in [("dict", s1["corrections"]), ("csc", s2["corrections"])]:
        for candidate in candidates:
            if source == "dict":
                start, end = candidate["position_in_source"]
            else:
                start = _map_s2(candidate["position"][0], offsets)
                if start is None:
                    continue
                end = start + 1
            span = strip_punct(aligner.gold_at(start, end))
            label = ("NOGOLD" if not span else "CORRECT" if span == strip_punct(candidate["to"])
                     else "SH" if span == strip_punct(candidate["from"]) else "WF")
            entry = entries.get(candidate.get("entry_id"), {})
            decision = decisions[candidate["diff_id"]]
            rows.append({"file": filename, "diff_id": candidate["diff_id"], "source": source,
                         "from": candidate["from"], "to": candidate["to"], "gold": span, "label": label,
                         "category": CATEGORY_MAP.get(entry.get("category"), "other"),
                         "applied": candidate["diff_id"] in accepted, "decision": decision["decision"],
                         "reason": decision.get("reason", ""), "source_start": start, "source_end": end})
    return rows


def category_stats(rows, nogold_policy):
    if nogold_policy not in {"paper", "separate"}:
        raise ValueError("Unknown unmatched-reference policy")
    stats = {}
    for category in CATEGORIES:
        proposed = [r for r in rows if r["category"] == category]
        accepted = [r for r in proposed if r["applied"]]
        labels = Counter(r["label"] for r in accepted)
        rejected = Counter(r["label"] for r in proposed if not r["applied"])
        correct = labels["CORRECT"] + (labels["NOGOLD"] if nogold_policy == "paper" else 0)
        stats[category] = {"proposed": len(proposed), "accepted": len(accepted), "correct": correct,
                           "harmful": labels["SH"], "ineffective": labels["WF"], "unmatched": labels["NOGOLD"],
                           "precision": correct / len(accepted) if accepted else None,
                           "harmful_proportion": labels["SH"] / len(accepted) if accepted else None,
                           "acceptance_rate": len(accepted) / len(proposed) if proposed else None,
                           "rejected_correct": rejected["CORRECT"], "rejected_harmful": rejected["SH"],
                           "rejected_unmatched": rejected["NOGOLD"]}
    return stats


def analyze(args):
    root = Path(args.output)
    require_complete(root)
    manifest = read_json(root / "manifest.json")
    condition = next(c for c in manifest["conditions"] if c["name"] == args.condition)
    if not condition["review"]:
        raise ValueError("Case attribution requires a reviewed candidate pipeline")
    require_resources("dictionary.json")
    entries = {e["id"]: e for e in read_json(RESOURCES / "dictionary.json")["entries"]}
    summaries = []
    for run in range(1, manifest["settings"]["repeats"] + 1):
        rows = []
        for name in manifest["files"]:
            proposals = root / args.condition / "proposals" / name
            rows.extend(classify(read_json(proposals / "step1/corrections.json"),
                                 read_json(proposals / "step2/corrections.json"),
                                 read_json(root / args.condition / f"run{run}" / name / "llm_decisions.json"),
                                 read_text(Path(args.gold_dir) / name), entries, name))
        out = root / "case_study" / args.condition / f"run{run}"
        write_json(out / "ops_classified.json", rows)
        if rows:
            with (out / "ops_classified.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        stats = category_stats(rows, args.nogold_policy)
        summaries.append(stats)
        write_json(out / "category_stats.json", stats)
    means = {cat: {key: (sum(values) / len(values) if values else None)
                   for key in summaries[0][cat]
                   for values in [[s[cat][key] for s in summaries if s[cat][key] is not None]]}
             for cat in CATEGORIES}
    write_json(root / "case_study" / args.condition / "summary.json", {
        "nogold_policy": args.nogold_policy, "repeats": len(summaries), "mean": means,
        "note": "Paper policy requires manual confirmation of accepted unmatched proposals; rejected unmatched stay separate."})
    print("Case statistics saved; detailed records contain private text spans")
