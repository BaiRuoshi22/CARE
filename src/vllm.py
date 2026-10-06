"""Offline vLLM inference, baseline prompts and response extraction."""
import json
import os
import re
import sys
import time
from pathlib import Path
from .utils.config import Condition
from .utils.io import RESOURCES, environment, read_json, read_text, require_resources, write_json, write_text
from .review import execute, parse_decisions


class VLLMBackend:
    def __init__(self, model_path, settings):
        model_path = Path(model_path).resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError("Download Qwen first; pass a local --qwen-model directory")
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        from vllm import LLM, SamplingParams
        self.settings = settings
        self.llm = LLM(model=str(model_path), dtype=settings["dtype"], tensor_parallel_size=1,
                       max_model_len=settings["max_model_len"], max_num_seqs=settings["max_num_seqs"],
                       gpu_memory_utilization=settings["gpu_memory_utilization"],
                       language_model_only=True, generation_config="vllm", seed=settings["seed"])
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = {limit: SamplingParams(temperature=settings["temperature"], max_tokens=limit,
                                               seed=settings["seed"])
                         for limit in {settings["max_tokens"], settings["generation_max_tokens"]}}
        import torch
        self.metadata = {"environment": environment(), "model_id": settings["model_id"],
                         "model_config": read_json(model_path / "config.json"),
                         "dtype_requested": settings["dtype"],
                         "dtype_resolved": str(self.llm.llm_engine.model_config.dtype),
                         "gpu_name": torch.cuda.get_device_name(0),
                         "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                         "settings": settings, "language_model_only": True, "generation_config": "vllm"}

    def generate(self, system, user, max_tokens=None):
        max_tokens = max_tokens or self.settings["max_tokens"]
        tokens = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(tokens) + max_tokens > self.settings["max_model_len"]:
            raise ValueError("Prompt exceeds configured context; inputs are never silently truncated")
        start = time.perf_counter()
        completion = self.llm.generate([{"prompt_token_ids": tokens}], self.sampling[max_tokens], use_tqdm=False)[0].outputs[0]
        metadata = {"prompt_tokens": len(tokens), "output_tokens": len(completion.token_ids),
                    "seconds": time.perf_counter() - start, "finish_reason": completion.finish_reason}
        return completion.text, metadata


LABEL = re.compile(r'^\*\*[^*\n]{1,20}?\*\*\s*[：:]\s*|^\*\*[^*\n]{1,20}?[：:]\s*\*\*\s*')

SECTION = re.compile(r'^\*\*?【[^】]*】\*\*?\s*$')

NOTE = re.compile(r'[（(]注[：:][^）)]*[）)]')

SEP = re.compile(r'^(-{3,}|\*{3,}|_{3,})$')

START_H = re.compile(r'^#{1,6}\s*.*(修正|纠正|校正|修改|订正)后的.*(对话|文本|内容|记录)')

START_B = re.compile(r'^\*\*[【]?.{0,24}(修正|纠正|校正|修改|订正)后.{0,24}[】]?\*\*\s*[:：]?\s*$')

START_P = re.compile(r'^.{0,80}(以下是|下面是).{0,60}(修正|纠正|校正|修改|订正|润色).{0,60}[:：]\s*$')

END_H = re.compile(
    r'^#{1,6}\s*.*('
    r'(修正|纠错|错误|修改|订正).*(说明|分析|对照|点|清单|列表)'
    r'|对照表|总结|小结|说明如下|供参考'
    r'|(整理|润色|流畅).*版'
    r')')

HEADING = re.compile(r'^#{1,6}\s')

META = re.compile(r'^\*\*(场景|人物|参与人员|参与者|背景|时间|地点|说明|备注|注)\*\*?\s*[：:]')

def clean(line):
    s = line.strip()
    if not s or SECTION.match(s) or HEADING.match(s) or META.match(s):
        return None
    s = LABEL.sub('', s).strip()
    if not s:
        return None
    s = NOTE.sub('', s)
    s = s.replace('**', '').replace('*', '')
    s = s.strip().strip('>').strip()
    return s or None

def find_bounds(lines):
    n = len(lines)
    head_zone = max(3, int(n * 0.30))

    start = 0
    for i in range(min(head_zone, n)):
        s = lines[i].strip()
        if SEP.match(s) or START_H.match(s) or START_B.match(s) or START_P.match(s):
            start = i + 1
    while start < n and not lines[start].strip():
        start += 1

    end = n - 1
    for i in range(start, n):
        s = lines[i].strip()
        if END_H.match(s):
            end = i - 1
            break
        # A separator followed closely by a closing heading ends the body before the separator.
        if SEP.match(s):
            for j in range(i + 1, min(i + 4, n)):
                t = lines[j].strip()
                if END_H.match(t):
                    end = i - 1
                    break
            else:
                continue
            break
    while end > start and not lines[end].strip():
        end -= 1
    return start + 1, end + 1


def _fmt_entry(e: dict) -> str:
    """Render one dictionary entry as a line: canonical (aliases): note."""
    parts = [e["canonical"]]
    al = [a for a in e.get("aliases", []) if a and a != e["canonical"]]
    if al:
        parts.append(f"（又称：{' / '.join(al)}）")
    note = (e.get("note") or "").strip()
    return f"- {''.join(parts)}" + (f"：{note}" if note else "")


def baseline_prompt(kind):
    if kind == "free":
        return read_json(RESOURCES / "prompts/templates.json")["free"]
    strong = read_text(RESOURCES / "prompts/conservative.txt")
    if kind == "conservative":
        return strong
    if kind != "dictionary":
        raise ValueError("Unknown baseline: " + kind)
    entries = sorted(read_json(RESOURCES / "dictionary.json")["entries"], key=lambda e: (e.get("priority", 9), e["id"]))
    return strong + read_text(RESOURCES / "prompts/dictionary_suffix.txt") + "\n".join(_fmt_entry(e) for e in entries) + "\n"


def extract_body(raw, override=None):
    lines = raw.split("\n")
    start, end = override if override is not None else find_bounds(lines)
    if not 1 <= start <= end <= len(lines):
        raise ValueError("Invalid free-generation body boundaries")
    body = [s for s in (clean(line) for line in lines[start - 1:end]) if s]
    if not body:
        raise ValueError("Free-generation body is empty")
    return "\n".join(body) + "\n", {"start_line": start, "end_line": end,
                                     "total_lines": len(lines), "manual_override": override is not None}


def review_user(diffs):
    prefix = read_json(RESOURCES / "prompts/templates.json")["review_user_prefix"]
    return prefix + json.dumps({"diffs": diffs}, ensure_ascii=False, indent=2)


def infer(args, backend_factory=None):
    root = Path(args.output)
    manifest = read_json(root / "manifest.json")
    if not manifest["prepared"]:
        raise ValueError("Proposal preparation is incomplete")
    if (root / "inference_status.json").exists():
        raise FileExistsError("Inference already attempted here; use a fresh run directory")
    settings = manifest["settings"]
    conditions = [Condition(**row) for row in manifest["conditions"]]
    required = set()
    for condition in conditions:
        if condition.review:
            required.update(["prompts/reviewer.txt", "prompts/templates.json"])
        if condition.generation == "free":
            required.add("prompts/templates.json")
        if condition.generation in {"conservative", "dictionary"}:
            required.add("prompts/conservative.txt")
        if condition.generation == "dictionary":
            required.update(["dictionary.json", "prompts/dictionary_suffix.txt"])
    require_resources(*required)
    status = {"complete": False, "failures": [], "completed_files": 0}
    write_json(root / "inference_status.json", status)
    backend = None
    if any(c.stochastic for c in conditions):
        if backend_factory is None:
            backend_factory = VLLMBackend
        backend = backend_factory(manifest["paths"]["qwen"], settings)
        write_json(root / "backend.json", backend.metadata)
    for condition in conditions:
        for run in range(1, (settings["repeats"] if condition.stochastic else 1) + 1):
            for name in manifest["files"]:
                out = root / condition.name / f"run{run}" / name
                try:
                    if condition.generation:
                        text = read_text(Path(manifest["paths"]["asr"]) / name)
                        raw, info = backend.generate(baseline_prompt(condition.generation), text,
                                                     max_tokens=settings["generation_max_tokens"])
                        write_text(out / "raw_response.txt", raw)
                        write_json(out / "generation.json", info)
                        if info["finish_reason"] != "stop":
                            raise ValueError("Generation did not complete normally")
                        if not raw.strip():
                            raise ValueError("Empty generation response")
                        if condition.generation == "free":
                            final, bounds = extract_body(raw)
                            write_json(out / "extraction.json", bounds)
                        else:
                            final = raw
                    else:
                        proposal = root / condition.name / "proposals" / name
                        s1 = read_json(proposal / "step1/corrections.json")
                        s2 = read_json(proposal / "step2/corrections.json")
                        diffs = read_json(proposal / "diffs.json")
                        if condition.review:
                            decisions = []
                            if diffs:
                                raw, info = backend.generate(read_text(RESOURCES / "prompts/reviewer.txt"), review_user(diffs))
                                write_text(out / "raw_response.txt", raw)
                                write_json(out / "generation.json", info)
                                if info["finish_reason"] != "stop":
                                    raise ValueError("Review did not complete normally")
                                decisions = parse_decisions(raw, diffs)
                            final, log = execute(s1, s2, decisions, allow_overlap=condition.mask == "none")
                            write_json(out / "llm_decisions.json", log)
                        else:
                            final = s2["result_text"]
                    write_text(out / "final.txt", final)
                    status["completed_files"] += 1
                    print(json.dumps({"condition": condition.name, "run": run, "file": name, "status": "ok"}), flush=True)
                except (ValueError, TypeError, KeyError) as error:
                    failure = {"condition": condition.name, "run": run, "file": name, "error_type": type(error).__name__}
                    status["failures"].append(failure)
                    write_json(out / "failure.json", failure)
                    print(json.dumps(failure), flush=True)
    status["complete"] = not status["failures"]
    write_json(root / "inference_status.json", status)
    if not status["complete"]:
        raise RuntimeError("Incomplete inference; inspect private failure files before evaluation")
