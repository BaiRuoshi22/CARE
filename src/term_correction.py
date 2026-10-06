"""Deterministic domain proposals; final experiment matching rules."""
import json
from collections import defaultdict
from pypinyin import lazy_pinyin, Style

def is_chinese_char(c: str) -> bool:
    return '\u4e00' <= c <= '\u9fff'

def is_chinese(s: str) -> bool:
    return bool(s) and all(is_chinese_char(c) for c in s)

def get_pinyin(text: str) -> str:
    return " ".join(lazy_pinyin(text, style=Style.NORMAL))

def make_fingerprint(pinyin_str: str, rules: dict) -> str:
    initial_conf = {k: v for k, v in rules["initial_confusions"].items()
                    if not k.startswith('_')}
    final_conf = {k: v for k, v in rules["final_confusions"].items()
                  if not k.startswith('_')}
    out = []
    for syl in pinyin_str.split():
        for src in sorted(initial_conf, key=lambda k: -len(k)):
            if syl.startswith(src):
                syl = initial_conf[src] + syl[len(src):]
                break
        for src in sorted(final_conf, key=lambda k: -len(k)):
            if syl.endswith(src):
                syl = syl[:-len(src)] + final_conf[src]
                break
        out.append(syl)
    return " ".join(out)

class DictCorrector:
    MIN_PHONETIC_MATCH_LEN = 2           # Minimum window for pinyin matching (v3: 3 -> 2).
    MIN_PHONETIC_TARGET_LEN = 2          # Minimum pinyin replacement target length (added in v3).
    CROSS_LENGTH_ENABLE = True           # Cross-length matching (added in v3).

    def __init__(self, dict_path: str, rules_path: str):
        with open(dict_path, encoding='utf-8') as f:
            self.dictionary = json.load(f)["entries"]
        with open(rules_path, encoding='utf-8') as f:
            self.rules = json.load(f)
        self._build_indexes()

    def _build_indexes(self):
        """Build the layered lookup indexes."""
        # Literal index (Chinese and Latin), stored in full for the Step2 protection mask.
        self.literal_index = {}
        # Pinyin index: Chinese words of length >= 2 only, so single characters never become targets.
        self.pinyin_index = {}
        self.fingerprint_index = defaultdict(list)
        # Buckets by syllable count, for cross-length matching.
        self.fp_idx_by_syl = defaultdict(lambda: defaultdict(list))
        # asr_misrecognitions index: misheard form -> (target, entry_id).
        # Bypasses pinyin rules for frequent ASR errors whose syllables differ too much for
        # fingerprint matching (e.g. 刀疤 -> 多巴胺, "dao ba" vs "duo ba an").
        self.asr_misrecog_index = {}
        # Lengths of misheard forms, used when sizing the sliding window.
        self.asr_misrecog_lengths = set()
        self.cn_lengths = set()

        for entry in self.dictionary:
            self._register_word(entry["canonical"], entry["id"])
        for entry in self.dictionary:
            for w in entry["aliases"]:
                self._register_word(w, entry["id"])

        for entry in self.dictionary:
            for err_form, correct_form in entry.get("asr_misrecognitions", {}).items():
                if err_form and correct_form:
                    self.asr_misrecog_index[err_form] = (correct_form, entry["id"])
                    if is_chinese(err_form):
                        self.asr_misrecog_lengths.add(len(err_form))
                        self.cn_lengths.add(len(err_form))

        self.max_len = max(self.cn_lengths) if self.cn_lengths else 6
        self.min_len = max(min(self.cn_lengths) if self.cn_lengths else 2, 1)
        self.max_len = min(self.max_len, 10)

    def _register_word(self, word: str, entry_id: str):
        if not word:
            return
        # Literal index covers every length, including L=1, for the Step2 protection mask.
        self.literal_index.setdefault(word, entry_id)
        if not is_chinese(word):
            return
        self.cn_lengths.add(len(word))
        # Pinyin/fp indexes only register words of length >= MIN_PHONETIC_TARGET_LEN,
        # so single-character aliases such as 伏, 右 and 无 never become pinyin targets.
        if len(word) < self.MIN_PHONETIC_TARGET_LEN:
            return
        py = get_pinyin(word)
        self.pinyin_index.setdefault(py, (word, entry_id))
        fp = make_fingerprint(py, self.rules)
        self.fingerprint_index[fp].append((word, entry_id, py))
        N = len(py.split())
        self.fp_idx_by_syl[N][fp].append((word, entry_id, py))

    def _weighted_edit_distance(self, a_py: str, b_py: str) -> float:
        a = a_py.split()
        b = b_py.split()
        n, m = len(a), len(b)
        if n == 0:
            return float(m)
        if m == 0:
            return float(n)
        weights = self.rules["edit_distance_weights"]
        w_initial = weights.get("initial_confused", 0.3)
        w_final = weights.get("final_confused", 0.3)
        w_diff = weights.get("completely_diff", 1.0)
        dp = [[0.0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = float(i)
        for j in range(m + 1):
            dp[0][j] = float(j)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if a[i - 1] == b[j - 1]:
                    cost = 0.0
                else:
                    fp_a = make_fingerprint(a[i - 1], self.rules)
                    fp_b = make_fingerprint(b[j - 1], self.rules)
                    if fp_a == fp_b:
                        cost = max(w_initial, w_final)
                    else:
                        cost = w_diff
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,
                    dp[i][j - 1] + 1.0,
                    dp[i - 1][j - 1] + cost,
                )
        return dp[n][m]

    def _match_window(self, text: str, source_pos: int, L: int,
                      max_ed_per_char: float):
        """Try to match text[source_pos:source_pos+L]."""
        window = text[source_pos: source_pos + L]
        if not is_chinese(window):
            return None

        # asr_misrecognitions first: frequent ASR misreadings listed in the dictionary
        # are replaced by their target directly, bypassing the pinyin rules.
        if window in self.asr_misrecog_index:
            target, entry_id = self.asr_misrecog_index[window]
            return ('replace', target, entry_id, 1.0, 'asr_misrecog', 0.0)

        if window in self.literal_index:
            return ('literal', self.literal_index[window], window)

        if L < self.MIN_PHONETIC_MATCH_LEN:
            return None

        window_py = get_pinyin(window)
        window_fp = make_fingerprint(window_py, self.rules)
        window_N = len(window_py.split())

        if window_py in self.pinyin_index:
            target, entry_id = self.pinyin_index[window_py]
            if target != window:
                return ('replace', target, entry_id, 1.0, 'exact_pinyin', 0.0)

        candidates = []
        if window_fp in self.fp_idx_by_syl.get(window_N, {}):
            for tgt, eid, tgt_py in self.fp_idx_by_syl[window_N][window_fp]:
                if tgt == window:
                    continue
                dist = self._weighted_edit_distance(window_py, tgt_py)
                candidates.append((dist, tgt, eid, tgt_py, window_N))

        if self.CROSS_LENGTH_ENABLE:
            for delta in (1, -1):
                target_N = window_N + delta
                if target_N <= 0 or target_N == window_N:
                    continue
                if window_fp in self.fp_idx_by_syl.get(target_N, {}):
                    for tgt, eid, tgt_py in self.fp_idx_by_syl[target_N][window_fp]:
                        dist = self._weighted_edit_distance(window_py, tgt_py)
                        candidates.append((dist, tgt, eid, tgt_py, target_N))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        best_dist, best_tgt, best_eid, _best_py, best_N = candidates[0]
        denom = max(L, best_N)
        if best_dist <= max_ed_per_char * denom:
            sim = max(0.0, 1.0 - best_dist / max(denom, 1))
            match_type = ('fingerprint+ed' if best_N == window_N
                          else 'cross_length+fp')
            return ('replace', best_tgt, best_eid, round(sim, 3), match_type,
                    round(best_dist, 3))
        return None

    def correct(self, text: str) -> dict:
        corrections = []
        protected_spans = []
        result_segments = []
        result_pos = 0
        source_pos = 0
        diff_counter = 0

        thresholds = self.rules["match_thresholds"]
        max_ed_per_char = thresholds.get("max_weighted_edit_per_char", 0.25)

        n = len(text)
        while source_pos < n:
            matched = False
            upper = min(self.max_len, n - source_pos)
            # Longer windows first.
            for L in range(upper, max(self.min_len, 1) - 1, -1):
                if L < 1:
                    break
                result = self._match_window(text, source_pos, L, max_ed_per_char)
                if result is None:
                    continue

                if result[0] == 'literal':
                    _, entry_id, window = result
                    result_segments.append(window)
                    if L >= 2:  # Single-character literals are not protected spans.
                        protected_spans.append({
                            "position_in_source": [source_pos, source_pos + L],
                            "position_in_result": [result_pos, result_pos + L],
                            "term": window,
                            "entry_id": entry_id,
                        })
                    result_pos += L
                    source_pos += L
                    matched = True
                    break

                # 'replace'
                _, target, entry_id, sim, match_type, ed = result
                window = text[source_pos: source_pos + L]
                diff_counter += 1
                corrections.append({
                    "diff_id": f"S1-{diff_counter:03d}",
                    "position_in_source": [source_pos, source_pos + L],
                    "position_in_result": [result_pos, result_pos + len(target)],
                    "from": window,
                    "to": target,
                    "match_type": match_type,
                    "entry_id": entry_id,
                    "matched_to": target,
                    "similarity": sim,
                    "edit_distance": ed,
                })
                result_segments.append(target)
                result_pos += len(target)
                source_pos += L
                matched = True
                break

            if not matched:
                result_segments.append(text[source_pos])
                result_pos += 1
                source_pos += 1

        return {
            "source_text": text,
            "result_text": ''.join(result_segments),
            "corrections": corrections,
            "protected_spans": protected_spans,
        }
