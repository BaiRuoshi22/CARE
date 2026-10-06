"""Proposal contexts, strict review validation and staged rollback execution."""
import json
import re
CONTEXT_WINDOW = 25

def extract_context(text: str, start: int, end: int, window: int = CONTEXT_WINDOW) -> str:
    """Return the text around [start, end), marking the edited span with brackets."""
    s = max(0, start - window)
    e = min(len(text), end + window)
    prefix = "…" if s > 0 else ""
    suffix = "…" if e < len(text) else ""
    return f"{prefix}{text[s:start]}「{text[start:end]}」{text[end:e]}{suffix}"

def build_diffs(step1_data: dict, step2_data: dict) -> list:
    """Merge Step1 and Step2 edits into one review list with context."""
    s1_text = step1_data['result_text']   # Also the Step2 input.
    s2_text = step2_data['result_text']
    diffs = []

    for c in step1_data['corrections']:
        s, e = c['position_in_result']
        ctx = extract_context(s1_text, s, e)
        diffs.append({
            "diff_id": c['diff_id'],
            "source": "dict",
            "from": c['from'],
            "to": c['to'],
            "context": ctx,
            "evidence": {
                "match_type": c['match_type'],
                "entry_id": c['entry_id'],
                "matched_to": c['matched_to'],
                "similarity": c.get('similarity'),
            }
        })

    for c in step2_data['corrections']:
        s, e = c['position']
        ctx = extract_context(s2_text, s, e)
        diffs.append({
            "diff_id": c['diff_id'],
            "source": "csc",
            "from": c['from'],
            "to": c['to'],
            "context": ctx,
            "evidence": {
                "match_type": "csc_mlm",
                "confidence_margin": c['confidence_margin'],
            }
        })
    return diffs

def _parse_json_response(content: str) -> dict:
    """Extract a JSON object from a response that may be wrapped in Markdown."""
    content = re.sub(r'```(?:json)?\s*', '', content).strip()
    content = content.rstrip('`').strip()
    s = content.find('{')
    e = content.rfind('}')
    if s < 0 or e < 0:
        raise ValueError(f"No JSON object found in response: {content[:200]}")
    return json.loads(content[s:e + 1])

def apply_decisions(step1_data: dict,
                    step2_data: dict,
                    decisions: list) -> tuple:
    """
    Start from the Step2 output and roll back every edit the reviewer rejected.

    Step2 protection keeps Step1 and Step2 edits from overlapping, so both groups
    of rollbacks can be applied to step2.result_text independently.

    Returns (final_text, applied_log, rollback_log).
    """
    decision_map = {d['diff_id']: d for d in decisions}

    final = step2_data['result_text']
    applied_log = []
    rollback_log = []

    # ---- Roll back Step2 first: single-character replacements of length 1 ----
    for c in step2_data['corrections']:
        d = decision_map[c['diff_id']]
        decision = d['decision']
        reason = d.get('reason', '')
        if decision == 'KEEP':
            applied_log.append(('S2', c, decision, reason))
        else:  # ROLLBACK or UNCERTAIN
            pos = c['position'][0]
            if pos < len(final) and final[pos] == c['to']:
                final = final[:pos] + c['from'] + final[pos + 1:]
                rollback_log.append(('S2', c, decision, reason))

    # ---- Then roll back Step1; lengths may change, so go right to left ----
    s1_corrs = list(step1_data['corrections'])
    s1_corrs.sort(key=lambda c: -c['position_in_result'][0])
    for c in s1_corrs:
        d = decision_map[c['diff_id']]
        decision = d['decision']
        reason = d.get('reason', '')
        if decision == 'KEEP':
            applied_log.append(('S1', c, decision, reason))
        else:
            s, e = c['position_in_result']
            if final[s:e] == c['to']:
                final = final[:s] + c['from'] + final[e:]
                rollback_log.append(('S1', c, decision, reason))

    return final, applied_log, rollback_log


def validate_decisions(decisions, diffs):
    if not isinstance(decisions, list):
        raise ValueError("Review must contain a decisions list")
    expected = {d["diff_id"] for d in diffs}
    if len(expected) != len(diffs):
        raise ValueError("Duplicate proposal IDs")
    seen = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            raise ValueError("Non-object decision")
        key = decision.get("diff_id")
        if not isinstance(key, str) or key not in expected or key in seen:
            raise ValueError("Unknown or duplicate decision ID")
        if decision.get("decision") not in ("KEEP", "ROLLBACK", "UNCERTAIN"):
            raise ValueError("Invalid decision label")
        seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete review response")
    return decisions


def parse_decisions(content, diffs):
    parsed = _parse_json_response(content)
    if not isinstance(parsed, dict):
        raise ValueError("Review response must be a JSON object")
    return validate_decisions(parsed.get("decisions"), diffs)


def execute(s1, s2, decisions, allow_overlap=False):
    diffs = build_diffs(s1, s2)
    validate_decisions(decisions, diffs)
    final, applied, rolled = apply_decisions(s1, s2, decisions)
    executed = {c["diff_id"] for _, c, _, _ in applied + rolled}
    skipped = [d["diff_id"] for d in diffs if d["diff_id"] not in executed]
    if skipped and not allow_overlap:
        raise ValueError("Unexecuted decisions in a protected pipeline: " + ", ".join(skipped))
    log = lambda rows: [{"tag": tag, "diff_id": c["diff_id"], "decision": decision, "reason": reason}
                        for tag, c, decision, reason in rows]
    return final, {"diffs": diffs, "decisions": decisions, "applied": log(applied),
                   "rolled_back": log(rolled), "skipped_rollbacks": skipped,
                   "overlap_policy": "legacy-stage-order" if allow_overlap else "protected"}
