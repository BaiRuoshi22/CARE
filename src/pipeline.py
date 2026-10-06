"""Prepare dictionary/ReLM proposals, protection masks and thresholded candidates."""
import json
import os
import re
from pathlib import Path
from .utils.config import Condition, load_settings, select_conditions
from .term_correction import DictCorrector
from .utils.io import RESOURCES, environment, read_json, read_text, require_resources, write_json, write_text
from .review import build_diffs


def build_protected_mask(text: str,
                         step1_data: dict,
                         dict_path: str,
                         rules: dict) -> list:
    """Build a bool list as long as text; True marks a protected position that CSC may not edit."""
    mask = [False] * len(text)

    for c in step1_data['corrections']:
        s, e = c['position_in_result']
        for i in range(s, min(e, len(text))):
            mask[i] = True

    for p in step1_data['protected_spans']:
        s, e = p['position_in_result']
        for i in range(s, min(e, len(text))):
            mask[i] = True

    # Rescan text for every literal dictionary term, not just Step1's own hits.
    with open(dict_path, encoding='utf-8') as f:
        dic_data = json.load(f)
    literal_terms = []
    for entry in dic_data["entries"]:
        for w in [entry["canonical"]] + entry["aliases"]:
            if w and len(w) >= 2:
                literal_terms.append(w)
    literal_terms.sort(key=lambda w: -len(w))
    for term in literal_terms:
        start = 0
        while True:
            pos = text.find(term, start)
            if pos < 0:
                break
            for i in range(pos, pos + len(term)):
                mask[i] = True
            start = pos + 1

    patterns = rules.get("protected_patterns", {})
    for name, pat in patterns.items():
        if name.startswith('_'):
            continue
        try:
            for m in re.finditer(pat, text):
                for i in range(m.start(), m.end()):
                    mask[i] = True
        except re.error:
            continue

    return mask


def derive(pool, mask, margin):
    """Same semantics as the original ablation_derive.derive."""
    text = pool["text"]
    kept = sorted([c for c in pool["candidates"]
                   if not mask[c["position"]] and c["margin"] >= margin],
                  key=lambda c: c["position"])
    chars = list(text)
    corrections = []
    for i, c in enumerate(kept, 1):
        p = c["position"]
        assert chars[p] == c["from"], f"Source character mismatch at position {p}: {chars[p]} vs {c['from']}"
        chars[p] = c["to"]
        corrections.append({
            "diff_id": f"S2-{i:03d}", "position": [p, p + 1],
            "from": c["from"], "to": c["to"],
            "match_type": "csc_mlm", "confidence_margin": c["margin"],
        })
    return "".join(chars), corrections


def make_mask(policy, text, stage1, rules):
    if policy == "full":
        return build_protected_mask(text, stage1, str(RESOURCES / "dictionary.json"), rules)
    if policy == "none":
        return [False] * len(text)
    if policy != "regex_only":
        raise ValueError("Unknown protection policy: " + policy)
    mask = [False] * len(text)
    for name, pattern in rules.get("protected_patterns", {}).items():
        if not name.startswith("_"):
            for match in re.finditer(pattern, text):
                mask[match.start():match.end()] = [True] * (match.end() - match.start())
    return mask


def prepare(args):
    settings = load_settings(args.config)
    if args.repeats is not None:
        if args.repeats < 1:
            raise ValueError("repeats must be positive")
        settings["repeats"] = args.repeats
    if args.files:
        if len(set(args.files)) != len(args.files):
            raise ValueError("Duplicate file IDs")
        settings["files"] = args.files
    conditions = [Condition(**row) for row in select_conditions(args.config, args.suite, args.conditions)]
    require_resources("dictionary.json", "phonetic_rules.json")
    paths = {"asr": str(Path(args.asr_dir).resolve()), "qwen": str(Path(args.qwen_model).resolve()),
             "bert": str(Path(args.bert_model).resolve()), "relm": str(Path(args.relm_weights).resolve())}
    files = [f"{number}.txt" for number in settings["files"]]
    texts = {name: read_text(Path(paths["asr"]) / name) for name in files}
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "settings": settings, "paths": paths,
                "conditions": [vars(c) for c in conditions], "files": files,
                "environment": environment(), "prepared": False}
    write_json(root / "manifest.json", manifest)
    rules = read_json(RESOURCES / "phonetic_rules.json")
    dictionary = DictCorrector(str(RESOURCES / "dictionary.json"), str(RESOURCES / "phonetic_rules.json"))
    csc = None
    if any(c.csc for c in conditions):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from .mlm import CSCCorrector
        csc = CSCCorrector(model_path=paths["bert"], relm_weights=paths["relm"], confidence_margin=0.0,
                           max_src_len=settings["relm_max_src_len"], context=settings["relm_context"],
                           phonetic_filter=settings["relm_phonetic_filter"], device=args.csc_device)
    counts = []
    for name, original in texts.items():
        stage1_cache, pools = {}, {}
        for condition in conditions:
            if condition.generation:
                continue
            use_dict = condition.dictionary
            if use_dict not in stage1_cache:
                stage1_cache[use_dict] = dictionary.correct(original) if use_dict else {
                    "source_text": original, "result_text": original, "corrections": [], "protected_spans": []}
            s1 = stage1_cache[use_dict]
            text = s1["result_text"]
            mask = make_mask(condition.mask, text, s1, rules)
            if condition.csc:
                if use_dict not in pools:
                    _, candidates = csc.correct(text, [False] * len(text))
                    pools[use_dict] = {"text": text, "candidates": [
                        {"position": c["position"][0], "from": c["from"], "to": c["to"],
                         "margin": c["confidence_margin"]} for c in candidates]}
                    write_json(root / "pools" / ("dict" if use_dict else "raw") / f"{name}.json", pools[use_dict])
                final, corrections = derive(pools[use_dict], mask, condition.margin)
            else:
                final, corrections = text, []
            s2 = {"source_text": text, "result_text": final, "corrections": corrections,
                  "confidence_margin": condition.margin, "protected_char_count": sum(mask),
                  "mask_policy": condition.mask, "backbone": "relm" if condition.csc else None}
            out = root / condition.name / "proposals" / name
            for stage, value in [("step1", s1), ("step2", s2)]:
                write_json(out / stage / "corrections.json", value)
                write_text(out / stage / "corrected.txt", value["result_text"])
            write_json(out / "diffs.json", build_diffs(s1, s2))
            counts.append({"condition": condition.name, "file": name,
                           "dictionary": len(s1["corrections"]), "csc": len(s2["corrections"])})
    manifest["prepared"] = True
    write_json(root / "manifest.json", manifest)
    write_json(root / "proposal_counts.json", counts)
    print(f"Prepared {len(files)} files for {len(conditions)} conditions: {root}")
