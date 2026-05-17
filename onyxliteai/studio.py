from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .config import ProjectConfig, load_project_config
from .generate import generate_reply, load_model
from .model import OnyxThornAI

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

BRAND = "onyxthornAI"
DEFAULT_CONFIG = "configs/onyxthornai_chat_17m.yaml"
SESSION_CONFIG = ".onyxthorn_studio_config.yaml"


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def wait_enter() -> None:
    input("\nEnter — назад в меню...")


def pick_auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_config(cfg: ProjectConfig, path: str | Path = SESSION_CONFIG) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required. Install: pip install -r requirements.txt")
    payload = {"model": asdict(cfg.model), "train": asdict(cfg.train)}
    p = Path(path)
    p.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def model_param_count(cfg: ProjectConfig, fallback_vocab: int = 16000) -> int:
    original = cfg.model.vocab_size
    if cfg.model.vocab_size <= 0:
        cfg.model.vocab_size = fallback_vocab
    try:
        return OnyxThornAI(cfg.model).parameter_count()
    finally:
        cfg.model.vocab_size = original


def estimate_seconds(cfg: ProjectConfig, params: int) -> float:
    device = cfg.train.device if cfg.train.device != "auto" else pick_auto_device()
    tokens_per_update = cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.model.block_size
    base_tokens = 4 * 12 * 1024
    base_params = 17_000_000
    # Conservative rough estimate. Real value depends heavily on GPU/CPU, RAM bandwidth and PyTorch build.
    if device == "cuda":
        base_sec = 0.55
    elif device == "mps":
        base_sec = 1.80
    else:
        base_sec = 8.00
    sec_per_step = base_sec * (params / base_params) * (tokens_per_update / base_tokens)
    return max(1.0, sec_per_step * cfg.train.max_steps)


def bytes_to_mb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 2)


def info_block(cfg: ProjectConfig) -> str:
    params = model_param_count(cfg)
    tokens_per_update = cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.model.block_size
    seconds = estimate_seconds(cfg, params)
    lines = [
        f"model: {BRAND}",
        f"params: {params:,}",
        f"context: {cfg.model.block_size} tokens",
        f"layers/heads/emb: {cfg.model.n_layer}/{cfg.model.n_head}/{cfg.model.n_embd}",
        f"dropout/bias: {cfg.model.dropout}/{cfg.model.bias}",
        f"vocab estimate: {cfg.model.vocab_size or 16000}",
        f"model size fp16/bf16: ≈{bytes_to_mb(params * 2):.1f} MB",
        f"model size fp32: ≈{bytes_to_mb(params * 4):.1f} MB",
        f"full training checkpoint with optimizer: roughly 3-5x fp32 weights",
        f"steps: {cfg.train.max_steps:,}",
        f"batch x accum: {cfg.train.batch_size} x {cfg.train.grad_accum_steps}",
        f"tokens/update: ≈{tokens_per_update:,}",
        f"device: {cfg.train.device} -> {pick_auto_device() if cfg.train.device == 'auto' else cfg.train.device}",
        f"rough training time: ≈{seconds:.0f} seconds",
        f"out_dir: {cfg.train.out_dir}",
        f"tokenizer: {cfg.train.tokenizer_path}",
    ]
    return "\n".join(lines)


def ask_int(label: str, current: int, min_value: int = 1) -> int:
    raw = input(f"{label} [{current}]: ").strip()
    if not raw:
        return current
    value = int(raw)
    if value < min_value:
        raise ValueError(f"{label} must be >= {min_value}")
    return value


def ask_float(label: str, current: float, min_value: float = 0.0) -> float:
    raw = input(f"{label} [{current}]: ").strip()
    if not raw:
        return current
    value = float(raw.replace(",", "."))
    if value < min_value:
        raise ValueError(f"{label} must be >= {min_value}")
    return value


def edit_training(cfg: ProjectConfig) -> None:
    print(color("Настройки обучения. Пустой ввод оставляет текущее значение.", "36"))
    cfg.train.max_steps = ask_int("steps", cfg.train.max_steps)
    cfg.train.batch_size = ask_int("batch_size", cfg.train.batch_size)
    cfg.train.grad_accum_steps = ask_int("grad_accum_steps", cfg.train.grad_accum_steps)
    cfg.train.eval_interval = ask_int("eval_interval", cfg.train.eval_interval)
    cfg.train.save_interval = ask_int("save_interval", cfg.train.save_interval)
    block_size = ask_int("context / block_size", cfg.model.block_size, 128)
    if block_size % 64 != 0:
        print("Лучше брать кратно 64; значение принято, но может быть неудобным.")
    cfg.model.block_size = block_size
    cfg.train.learning_rate = ask_float("learning_rate", cfg.train.learning_rate)
    save_config(cfg)
    print(f"\nSaved temporary config: {SESSION_CONFIG}")
    print(info_block(cfg))


def jsonl_inputs() -> list[str]:
    patterns = ["data/seed/*.jsonl", "data/generated/*.jsonl", "data/raw/*.jsonl"]
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))
    return sorted(set(files))


def prepare_data(cfg: ProjectConfig) -> None:
    files = jsonl_inputs()
    if not files:
        print("JSONL files not found.")
        return
    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--jsonl",
        *files,
        "--tokenizer",
        cfg.train.tokenizer_path,
        "--out_dir",
        cfg.train.data_dir,
        "--corpus",
        "data/processed/tokenizer_corpus.txt",
        "--vocab_size",
        str(cfg.model.vocab_size or 16000),
    ]
    print(color("Запускаю подготовку данных:", "36"), " ".join(cmd[:4]), "...")
    subprocess.run(cmd, check=True)


def train(cfg: ProjectConfig) -> None:
    save_config(cfg)
    print(info_block(cfg))
    ok = input("\nНачать обучение с этими настройками? [y/N]: ").strip().lower()
    if ok != "y":
        return
    cmd = [sys.executable, "-m", "onyxliteai.train", "--config", SESSION_CONFIG]
    subprocess.run(cmd, check=True)


def find_default_checkpoint(cfg: ProjectConfig) -> str:
    out = Path(cfg.train.out_dir)
    candidates = [
        out / "best_infer_fp16.pt",
        out / "final_infer_fp16.pt",
        out / "best.pt",
        out / "final.pt",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return str(out / "best.pt")


def chat(cfg: ProjectConfig) -> None:
    checkpoint = input(f"Checkpoint [{find_default_checkpoint(cfg)}]: ").strip() or find_default_checkpoint(cfg)
    tokenizer_path = input(f"Tokenizer [{cfg.train.tokenizer_path}]: ").strip() or cfg.train.tokenizer_path
    device = input(f"Device [auto]: ").strip() or "auto"
    temperature = 0.70
    top_k = 50
    top_p = 0.92
    max_new_tokens = 256
    repetition_penalty = 1.08

    model, tokenizer, actual_device = load_model(checkpoint, tokenizer_path, device)
    print(color(f"\n{BRAND} chat loaded. Commands: /exit, /info, /settings, /reload, /reset", "32"))
    while True:
        user = input("\nYou> ").strip()
        if user.lower() in {"/exit", "exit", "quit", "q"}:
            return
        if user.lower() == "/info":
            print(
                f"{BRAND} | params={model.parameter_count():,} | context={model.config.block_size} | "
                f"vocab={model.config.vocab_size} | device={actual_device}"
            )
            continue
        if user.lower() == "/reset":
            print("Контекст сброшен. Текущая версия работает без истории между сообщениями.")
            continue
        if user.lower() == "/reload":
            model, tokenizer, actual_device = load_model(checkpoint, tokenizer_path, device)
            print("Модель перезагружена.")
            continue
        if user.lower() == "/settings":
            print("Пустой ввод оставляет значение.")
            temperature = ask_float("temperature", temperature, 0.01)
            top_k = ask_int("top_k", top_k, 0)
            top_p = ask_float("top_p", top_p, 0.0)
            max_new_tokens = ask_int("max_new_tokens", max_new_tokens)
            repetition_penalty = ask_float("repetition_penalty", repetition_penalty, 1.0)
            continue
        started = time.time()
        answer = generate_reply(
            model=model,
            tokenizer=tokenizer,
            device=actual_device,
            user_message=user,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            use_fallback=True,
        )
        print(f"\n{BRAND}> {answer}")
        print(color(f"generated in {time.time() - started:.1f}s", "90"))


def main() -> None:
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    cfg = load_project_config(cfg_path)
    while True:
        clear()
        print(color("╔══════════════════════════════════════╗", "35"))
        print(color("║          onyxthornAI Studio          ║", "35"))
        print(color("╚══════════════════════════════════════╝", "35"))
        print(info_block(cfg))
        print("\n1 — изменить steps/batch/context")
        print("2 — подготовить данные и tokenizer")
        print("3 — начать обучение")
        print("4 — чат с обученной моделью")
        print("5 — сохранить текущий config")
        print("0 — выход")
        choice = input("\nВыбор: ").strip()
        try:
            if choice == "1":
                edit_training(cfg)
                wait_enter()
            elif choice == "2":
                prepare_data(cfg)
                wait_enter()
            elif choice == "3":
                train(cfg)
                wait_enter()
            elif choice == "4":
                chat(cfg)
                wait_enter()
            elif choice == "5":
                p = save_config(cfg)
                print(f"Saved: {p}")
                wait_enter()
            elif choice == "0":
                return
        except Exception as exc:
            print(color(f"Ошибка: {exc}", "31"))
            wait_enter()


if __name__ == "__main__":
    main()
