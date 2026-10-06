"""Text normalization and per-dialogue CER, edit-block, CCDR and touch metrics."""
import re
from difflib import SequenceMatcher
from rapidfuzz.distance import Levenshtein
lev_distance = Levenshtein.distance

_PUNCT_RE = re.compile(
    r"[\s"                             # All Unicode whitespace.
    r"﻿"                          # BOM. U+FEFF is neither in \s nor in any range below,
                                       # so it used to be compared as a content character. 10 of
                                       # the 30 ASR files carry a BOM and no reference does: the
                                       # pipeline keeps it and is charged one extra error, while any
                                       # rewriting output that drops it gains a free CORRECT,
                                       # systematically favoring the generative arms.
    r"\u3000-\u303F"                   # CJK symbols and punctuation.
    r"\uFF00-\uFFEF"                   # Fullwidth punctuation and symbols.
    r"\u2000-\u206F"                   # General punctuation.
    r"\u2E00-\u2E7F"                   # Supplemental punctuation.
    r"!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~"  # ASCII punctuation.
    r"…—–"                             # Ellipsis and dashes.
    r"]+"
)

def strip_punct_and_whitespace(text: str) -> str:
    """Remove all punctuation and whitespace, keeping only content characters."""
    return _PUNCT_RE.sub("", text)

def compute_cer(hyp: str, ref: str) -> dict:
    if not ref:
        return {"cer": 0.0, "edit_dist": 0, "N": 0}
    d = lev_distance(hyp, ref)
    return {"cer": d / len(ref), "edit_dist": d, "N": len(ref)}

def _mark_correct(a: str, b: str) -> list:
    ok = [False] * len(a)
    for tag, i1, i2, _, _ in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            for i in range(i1, i2): ok[i] = True
    return ok

def _mark_has_gold(a: str, gold: str) -> list:
    """True = the reference expects content at this position of a (equal/replace).
    False = the reference has nothing here (delete, i.e. spurious content in a)."""
    has = [False] * len(a)
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, gold, autojunk=False).get_opcodes():
        if tag in ("equal", "replace"):
            for i in range(i1, i2): has[i] = True
    return has

def compute_char_her(asr: str, corrected: str, gold: str) -> dict:
    """
    SH := destroyed correct content or content absent from the reference

    replace block  standard was_ok / now_ok
    delete block   B1: now_ok asks whether the reference expects content here
    insert block   B2 (asymmetric): was_ok asks whether the reference expects content here
    """
    asr_ok    = _mark_correct(asr, gold)
    corr_ok   = _mark_correct(corrected, gold)
    asr_ghas  = _mark_has_gold(asr, gold)
    corr_ghas = _mark_has_gold(corrected, gold)

    n_correct = n_sh = n_wf = n_neutral = 0
    details = []

    for tag, i1, i2, j1, j2 in SequenceMatcher(None, asr, corrected, autojunk=False).get_opcodes():
        if tag == "equal":
            continue

        before = asr[i1:i2]
        after  = corrected[j1:j2]

        if i2 > i1 and j2 > j1:
            # ---- REPLACE ----
            of = sum(asr_ok[i]  for i in range(i1, i2)) / (i2 - i1)
            cf = sum(corr_ok[j] for j in range(j1, j2)) / (j2 - j1)
            was_ok = of > 0.5
            now_ok = cf > 0.5

        elif j1 == j2:
            # ---- DELETE (B1) ----
            of = sum(asr_ok[i] for i in range(i1, i2)) / (i2 - i1)
            was_ok = of > 0.5
            gf = sum(asr_ghas[i] for i in range(i1, i2)) / (i2 - i1)
            now_ok = gf <= 0.5          # The reference has nothing here either, so the deletion is right.

        else:
            # ---- INSERT (B2, asymmetric) ----
            cf = sum(corr_ok[j] for j in range(j1, j2)) / (j2 - j1)
            now_ok = cf > 0.5
            gf = sum(corr_ghas[j] for j in range(j1, j2)) / (j2 - j1)
            was_ok = gf <= 0.5          # The reference has nothing here either, so the original absence was right.

        if   was_ok and not now_ok:     label = "SH";       n_sh += 1
        elif not was_ok and now_ok:     label = "CORRECT";  n_correct += 1
        elif not was_ok and not now_ok: label = "WF";       n_wf += 1
        else:                           label = "NEUTRAL";  n_neutral += 1

        details.append({
            "type": "del" if j1 == j2 else ("ins" if i1 == i2 else "rep"),
            "before": before[:50], "after": after[:50], "label": label,
        })

    total = n_correct + n_sh + n_wf + n_neutral
    return {
        "total_ops": total,
        "correct": n_correct, "sh": n_sh, "wf": n_wf, "neutral": n_neutral,
        "her": n_sh / total if total else 0.0,
        "cp":  n_correct / total if total else 0.0,
        "wfr": n_wf / total if total else 0.0,
        "details": details,
    }

def compute_ccdr(asr: str, corrected: str, gold: str) -> dict:
    asr_ok = _mark_correct(asr, gold)
    Hb = sum(asr_ok)
    destroyed = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, asr, corrected, autojunk=False).get_opcodes():
        if tag == "equal": continue
        destroyed += sum(asr_ok[i] for i in range(i1, i2))
    return {"ccdr": destroyed / Hb if Hb > 0 else 0.0, "destroyed": destroyed, "Hb": Hb}


def touch_rate(asr: str, out: str) -> dict:
    """Share of ASR characters touched: ASR characters inside non-equal blocks / all ASR characters."""
    touched = 0
    ins_blocks = 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, asr, out, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        touched += (i2 - i1)
        if i1 == i2:
            ins_blocks += 1
    return {"touched": touched, "total": len(asr),
            "tr": touched / len(asr) if asr else 0.0,
            "insert_blocks": ins_blocks}

def measure(asr_raw, gold_raw, out_raw):
    a = strip_punct_and_whitespace(asr_raw)
    g = strip_punct_and_whitespace(gold_raw)
    o = strip_punct_and_whitespace(out_raw)
    cb = compute_cer(a, g)
    ca = compute_cer(o, g)
    ch = compute_char_her(a, o, g)
    ei = compute_ccdr(a, o, g)
    tr = touch_rate(a, o)
    return {"cer_before": cb["cer"], "cer_after": ca["cer"],
            "cer_delta": ca["cer"] - cb["cer"],
            "edit_dist_before": cb["edit_dist"], "edit_dist_after": ca["edit_dist"],
            "N_gold": cb["N"], "len_asr": len(a), "len_out": len(o),
            "sh": ch["sh"], "correct": ch["correct"], "wf": ch["wf"],
            "neutral": ch["neutral"], "total_ops": ch["total_ops"],
            "her": ch["her"], "cp": ch["cp"], "wfr": ch["wfr"],
            "destroyed": ei["destroyed"], "Hb": ei["Hb"], "ccdr": ei["ccdr"],
            "touched": tr["touched"], "touch_rate": tr["tr"],
            "details": ch["details"]}
