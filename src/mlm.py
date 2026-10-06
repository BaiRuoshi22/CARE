"""ReLM inference with the original character alignment and window semantics."""
from pathlib import Path
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
BACKBONE_NAME = "relm"
def resolve_base_model_path(path):
    if not path or not Path(path).is_dir():
        raise FileNotFoundError("Download BERT first and pass its local directory")
    return str(path)
def resolve_relm_weights(path):
    if not path or not Path(path).is_file():
        raise FileNotFoundError("Download relm-m0.3.bin first and pass its local path")
    return str(path)
def _torch_load_state_dict(path):
    return torch.load(path, map_location="cpu", weights_only=True)

SENT_BOUNDARY = '。！？!?\n；;'

RELM_MAX_SRC_LEN = 254

_FUZZY_INITIALS = (("zh", "z"), ("ch", "c"), ("sh", "s"))

_FUZZY_FINALS = (("ang", "an"), ("eng", "en"), ("ing", "in"))

def _is_cjk(ch: str) -> bool:
    return '\u4e00' <= ch <= '\u9fff'

def fuzzy_pinyin(py: str) -> str:
    """Normalize toneless pinyin to common fuzzy sounds: zh/z ch/c sh/s, l/n, ang/an eng/en ing/in."""
    for a, b in _FUZZY_INITIALS:
        if py.startswith(a):
            py = b + py[len(a):]
            break
    if py.startswith("l"):
        py = "n" + py[1:]
    for a, b in _FUZZY_FINALS:
        if py.endswith(a):
            py = py[:-len(a)] + b
            break
    return py

class PhoneticFilter:
    """
    Optional pronunciation-consistency filter (off by default; off matches the original behavior):
      off   : no filtering
      exact : source and target share at least one toneless reading (heteronyms included)
      fuzzy : exact, plus common fuzzy-sound equivalences
    Requires pypinyin.
    """

    def __init__(self, mode: str = "off"):
        if mode not in ("off", "exact", "fuzzy"):
            raise ValueError(f"phonetic_filter must be off/exact/fuzzy, got {mode!r}")
        self.mode = mode
        self._cache = {}
        if mode != "off":
            try:
                from pypinyin import pinyin, Style
            except ImportError as e:
                raise ImportError("phonetic_filter requires pypinyin: pip install pypinyin") from e
            self._pinyin = pinyin
            self._style = Style.NORMAL

    def _readings(self, ch: str) -> set:
        if ch not in self._cache:
            r = self._pinyin(ch, style=self._style, heteronym=True)
            self._cache[ch] = set(r[0]) if r else set()
        return self._cache[ch]

    def allows(self, orig: str, pred: str) -> bool:
        if self.mode == "off":
            return True
        ra, rb = self._readings(orig), self._readings(pred)
        if self.mode == "exact":
            return bool(ra & rb)
        return bool({fuzzy_pinyin(x) for x in ra} & {fuzzy_pinyin(x) for x in rb})

def collect_edits(text, candidates, protected_mask, confidence_margin, phonetic_filter=None):
    """
    Filter per-position model candidates into an edit list using the original Step2 rules.
    candidates: iterable of dict(pos, pred, margin, p_orig), where pos is a character index in text.
    Rules (unchanged from the original): skip protected positions; the source character must be
    a Chinese character; the prediction must be a single, different Chinese character; and
    margin >= confidence_margin. An optional phonetic filter may also apply.
    """
    errors = []
    for c in candidates:
        pos = c["pos"]
        if pos < 0 or pos >= len(text):
            continue
        if protected_mask[pos]:
            continue
        orig = text[pos]
        if not _is_cjk(orig):
            continue
        pred = c["pred"]
        if (not pred or pred.startswith('[') or pred.startswith('#')
                or len(pred) != 1 or not _is_cjk(pred)):
            continue
        if pred == orig or c["margin"] < confidence_margin:
            continue
        if phonetic_filter is not None and not phonetic_filter.allows(orig, pred):
            continue
        errors.append({
            "position": pos,
            "from": orig,
            "to": pred,
            "margin": round(float(c["margin"]), 4),
            "p_orig": round(float(c.get("p_orig", 0.0)), 4),
        })
    return errors

def build_relm_input(src_ids, cls_id, sep_id, mask_id):
    """[CLS] src [SEP] [MASK]*n [SEP], matching the official DataProcessorForRephrasing inference input."""
    n = len(src_ids)
    return [cls_id] + list(src_ids) + [sep_id] + [mask_id] * n + [sep_id]

def relm_slot_range(n):
    """Index range [start, end) of the n rephrasing slots in the input sequence."""
    return n + 2, 2 * n + 2

def plan_windows(n, window, context):
    """
    Split a length-n sequence into windows: [(win_start, win_end, core_start, core_end), ...].
    The core ranges cover [0, n) exactly with no overlap; except at the ends, each core has
    `context` characters of context on both sides.
    """
    if n <= window:
        return [(0, n, 0, n)]
    if window - 2 * context <= 0:
        raise ValueError("window must be greater than 2*context")
    plans = []
    win_start, core_start = 0, 0
    while True:
        win_end = min(n, win_start + window)
        core_end = n if win_end == n else win_end - context
        plans.append((win_start, win_end, core_start, core_end))
        if core_end >= n:
            break
        core_start = core_end
        win_start = core_start - context
    return plans

def split_sentences(text):
    """Split at sentence-final punctuation and newlines; long sentences go to the sliding window."""
    segments, cur = [], 0
    for i, ch in enumerate(text):
        if ch in SENT_BOUNDARY:
            segments.append((cur, i + 1))
            cur = i + 1
    if cur < len(text):
        segments.append((cur, len(text)))
    return segments

class CSCCorrector:
    def __init__(self, model_path: str = None, confidence_margin: float = 0.85,
                 relm_weights: str = None, max_src_len: int = 62, context: int = 10,
                 phonetic_filter: str = "off", device: str = None):
        if not 1 <= max_src_len <= RELM_MAX_SRC_LEN:
            raise ValueError(f"max_src_len must be within 1..{RELM_MAX_SRC_LEN}")
        if max_src_len - 2 * context <= 0:
            raise ValueError("max_src_len must be greater than 2*context")

        base = resolve_base_model_path(model_path)
        weights = resolve_relm_weights(relm_weights)
        print(f"[Step2-ReLM] Loading tokenizer/backbone from: {base}")
        self.tokenizer = BertTokenizerFast.from_pretrained(base)
        self.model = BertForMaskedLM.from_pretrained(base)
        print(f"[Step2-ReLM] Loading ReLM weights from: {weights}")
        self._load_relm_weights(weights)
        self.model.eval()
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.backbone = BACKBONE_NAME
        self.confidence_margin = confidence_margin
        self.max_src_len = max_src_len
        self.context = context
        self.phonetic_filter = PhoneticFilter(phonetic_filter)
        self._char_id_cache = {}
        print(f"[Step2-ReLM] Model ready: device={self.device}, confidence_margin={confidence_margin}, "
              f"max_src_len={max_src_len}, context={context}, phonetic_filter={phonetic_filter}")

    def _load_relm_weights(self, path: str):
        sd = _torch_load_state_dict(path)
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
        if not isinstance(sd, dict):
            raise RuntimeError(f"{path} is not a state_dict (type {type(sd)})")
        sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}

        result = self.model.load_state_dict(sd, strict=False)
        benign = ("position_ids", "predictions.decoder.bias", "predictions.decoder.weight")
        missing = [k for k in result.missing_keys if not k.endswith(benign)]
        unexpected = [k for k in result.unexpected_keys if not k.endswith(benign)]
        loaded = len(sd) - len(result.unexpected_keys)
        print(f"[Step2-ReLM] Loaded {loaded} tensors; unexpected missing={len(missing)}, "
              f"unexpected={len(unexpected)}")
        if missing:
            print(f"  missing (first 10): {missing[:10]}")
        if unexpected:
            print(f"  unexpected (first 10): {unexpected[:10]}")
        if len(missing) > 4 or loaded < 100:
            raise RuntimeError("ReLM weights do not match BertForMaskedLM; check that relm-m0.3.bin was downloaded.")

    def _char_to_id(self, ch: str) -> int:
        tid = self._char_id_cache.get(ch)
        if tid is None:
            toks = self.tokenizer.tokenize(ch)
            tid = (self.tokenizer.convert_tokens_to_ids(toks[0]) if len(toks) == 1
                   else self.tokenizer.unk_token_id)
            self._char_id_cache[ch] = tid
        return tid

    @torch.no_grad()
    def _rephrase(self, src_ids):
        """Rephrase one window; return (top1_token, margin, p_orig) for each slot."""
        tok = self.tokenizer
        n = len(src_ids)
        input_ids = build_relm_input(src_ids, tok.cls_token_id, tok.sep_token_id, tok.mask_token_id)
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        attn = torch.ones_like(ids)
        logits = self.model(input_ids=ids, attention_mask=attn).logits[0]
        s, e = relm_slot_range(n)
        probs = torch.softmax(logits[s:e].float(), dim=-1)
        top2 = probs.topk(2, dim=-1)
        margins = (top2.values[:, 0] - top2.values[:, 1]).tolist()
        src_t = torch.tensor(src_ids, dtype=torch.long, device=probs.device)
        p_orig = probs.gather(1, src_t.unsqueeze(1)).squeeze(1).tolist()
        tokens = tok.convert_ids_to_tokens(top2.indices[:, 0].tolist())
        return list(zip(tokens, margins, p_orig))

    def correct_segment(self, text: str, protected_mask: list):
        """
        Run ReLM rephrasing correction on one sentence.
        Returns (corrected_text, errors_list), errors_list: [{position, from, to, margin, p_orig}, ...]
        """
        if not text or not text.strip():
            return text, []
        assert len(protected_mask) == len(text), \
            f"mask length {len(protected_mask)} != text length {len(text)}"

        # Whitespace is not sent to the model (BERT tokenization drops it); every other character is one token.
        positions = [i for i, ch in enumerate(text) if not ch.isspace()]
        if not positions:
            return text, []
        src_ids = [self._char_to_id(text[i]) for i in positions]

        candidates = []
        for ws, we, cs, ce in plan_windows(len(positions), self.max_src_len, self.context):
            preds = self._rephrase(src_ids[ws:we])
            for k in range(cs, ce):
                pred_tok, margin, p_orig = preds[k - ws]
                candidates.append({"pos": positions[k], "pred": pred_tok,
                                   "margin": margin, "p_orig": p_orig})

        errors = collect_edits(text, candidates, protected_mask,
                               self.confidence_margin, self.phonetic_filter)
        result_chars = list(text)
        for err in errors:
            result_chars[err["position"]] = err["to"]
        return ''.join(result_chars), errors

    def correct(self, text: str, protected_mask: list):
        """Process the full text sentence by sentence; long sentences use the sliding window in correct_segment."""
        result_chars = list(text)
        all_errors = []
        diff_counter = 0
        for (s, e) in split_sentences(text):
            _, errs = self.correct_segment(text[s:e], protected_mask[s:e])
            for err in errs:
                global_pos = s + err['position']
                diff_counter += 1
                all_errors.append({
                    "diff_id": f"S2-{diff_counter:03d}",
                    "position": [global_pos, global_pos + 1],
                    "from": err['from'],
                    "to": err['to'],
                    "match_type": "csc_mlm",
                    "confidence_margin": err['margin'],
                    "p_orig": err['p_orig'],
                    "backbone": self.backbone,
                })
                result_chars[global_pos] = err['to']
        return ''.join(result_chars), all_errors
