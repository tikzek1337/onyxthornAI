from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class ModelConfig:
    vocab_size: int = 0
    block_size: int = 512
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.10
    bias: bool = False


@dataclass
class TrainConfig:
    data_dir: str = "data/processed"
    tokenizer_path: str = "tokenizer/onyxliteai_tokenizer.json"
    out_dir: str = "runs/onyxliteai_chat_tiny"
    device: str = "auto"
    batch_size: int = 8
    grad_accum_steps: int = 8
    max_steps: int = 5000
    eval_interval: int = 250
    eval_iters: int = 50
    save_interval: int = 500
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    amp: str = "bf16"  # off | fp16 | bf16
    compile_model: bool = False
    seed: int = 1337
    resume: str = ""


@dataclass
class ProjectConfig:
    model: ModelConfig
    train: TrainConfig


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_project_config(path: str | Path | None = None) -> ProjectConfig:
    base = {"model": asdict(ModelConfig()), "train": asdict(TrainConfig())}
    if path:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load .yaml configs. Install requirements.txt first.")
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _deep_update(base, payload)
    return ProjectConfig(model=ModelConfig(**base["model"]), train=TrainConfig(**base["train"]))
