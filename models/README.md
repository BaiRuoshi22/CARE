# Models

Model checkpoints are not included in this repository. Download them separately before running inference.

## Required Components

| Component | Source | Local Path |
| --- | --- | --- |
| Qwen reviewer | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B), [Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | `models/Qwen3.6-35B-A3B/` |
| BERT backbone | [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese) | `models/bert-base-chinese/` |
| ReLM checkpoint | [Official release](https://drive.google.com/file/d/10vvkG_jzNK-CjIwlSvizhE1IOpnn9OqN/view) | `models/relm-m0.3.bin` |

ReLM is from Liu, Wu and Zhao, "Chinese Spelling Correction as Rephrasing Language Model," AAAI 2024.

Reserve approximately 70 GiB disk space for model weights and cache.
