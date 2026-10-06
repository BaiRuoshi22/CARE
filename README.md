# CARE

Candidate-constrained review for Chinese clinical ASR post-correction.

CARE combines dictionary correction (A), masked language model proposals (B), and LLM-based review (C). Stage A applies deterministic term correction; stage B generates character-level proposals via ReLM; stage C reviews candidates through Qwen with constrained prompts. The executor rolls back proposals rejected by the reviewer.

## Installation

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Model inference requires GPU support. For evaluation only:

```bash
python -m pip install pypinyin rapidfuzz numpy matplotlib
```

## Models and Data

Models are not included in this repository. Download them from public sources listed in [models/README.md](models/README.md).

The clinical evaluation corpus and dictionary resources are controlled-access materials. Contact the authors for:
- Clinical ASR data and reference transcripts
- Dictionary and prompt resources

See [data/README.md](data/README.md) for data format requirements.

## Usage

After obtaining the required models and data:

```bash
python -m src run --conditions care --repeats 1 \
  --asr-dir /path/to/asr --gold-dir /path/to/gold \
  --output outputs/trial
```

Run stages separately for debugging:

```bash
python -m src prepare --suite main --asr-dir /path/to/asr --output outputs/staged
python -m src infer --output outputs/staged
python -m src evaluate --output outputs/staged --gold-dir /path/to/gold
```

Available conditions include `care` (full pipeline), `free` (unconstrained generation), `conservative` (constrained generation), `dictionary` (dictionary-guided generation), and ablation variants. Evaluation computes normalized CER, character-level precision/recall, and touch rate.

## Repository Structure

```
src/                        Core implementation
  cli.py                   Command interface
  term_correction.py       Dictionary-based correction
  mlm.py                   Masked language model inference
  pipeline.py              Candidate preparation and filtering
  review.py                Review and decision logic
  vllm.py                  LLM interface
  metrics/                 Evaluation metrics
  analysis/                Analysis utilities
  utils/                   Shared utilities
configs/                   Configuration files
```
