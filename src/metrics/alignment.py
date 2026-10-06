"""Raw-coordinate reference alignment for proposal-level case studies."""
import re
from difflib import SequenceMatcher

_PUNCT_RE = re.compile(
    r"[\s　-〿＀-￯ -⁯⸀-⹿"
    r"!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~…—–﻿]+"
)

def strip_punct(text: str) -> str:
    return _PUNCT_RE.sub("", text)

class GoldAligner:
    """Map a span of the raw ASR text to the corresponding reference text."""

    def __init__(self, asr_raw: str, gold_raw: str):
        self.asr = asr_raw
        self.gold = gold_raw
        self.ops = SequenceMatcher(None, asr_raw, gold_raw, autojunk=False).get_opcodes()

    def gold_at(self, ss: int, se: int) -> str:
        parts = []
        for tag, i1, i2, j1, j2 in self.ops:
            os_, oe_ = max(i1, ss), min(i2, se)
            if os_ >= oe_:
                continue
            if tag == "equal":
                o = os_ - i1
                parts.append(self.gold[j1 + o: j1 + o + (oe_ - os_)])
            elif tag == "replace":
                hl, rl = i2 - i1, j2 - j1
                if hl > 0:
                    parts.append(self.gold[j1 + int((os_ - i1) / hl * rl):
                                           j1 + int((oe_ - i1) / hl * rl)])
            # Delete blocks have no reference content here and contribute an empty string.
        return "".join(parts)

def _s1_offset_table(s1_all):
    t = []
    for c in sorted(s1_all, key=lambda x: x["position_in_source"][0]):
        ss, se = c["position_in_source"]
        rs, re_ = c["position_in_result"]
        t.append((rs, re_, (re_ - rs) - (se - ss)))
    return t

def _map_s2(pos, ot):
    cum = 0
    for rs, re_, d in ot:
        if pos < rs:
            return pos - cum
        if pos < re_:
            return None          # Inside a Step1 edit; Step2 protection should prevent this.
        cum += d
    return pos - cum
