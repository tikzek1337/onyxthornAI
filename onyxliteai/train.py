from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch

from .config import ModelConfig, load_project_config
from .data import PackedTokenDataset
from .model import OnyxLiteAI, build_checkpoint
from .tokenizer import OnyxTokenizer


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_lr(step: int, max_steps: int, warmup_steps: int, learning_rate: float, min_lr: float) -> float:
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model: OnyxLiteAI, train_data: PackedTokenDataset, val_data: PackedTokenDataset, eval_iters: int, batch_size: int) -> dict:
    model.eval()
    out = {}
    for split, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train onyxLiteAI chat model")
    parser.add_argument("--config", type=str, default="configs/onyxliteai_chat_tiny.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt")
    parser.add_argument("--max_steps", type=int, default=None, help="Override train.max_steps from YAML for quick tests")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    if args.resume:
        cfg.train.resume = args.resume
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max_steps must be a positive integer")
        cfg.train.max_steps = args.max_steps
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, max(1, cfg.train.max_steps // 10))

    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = OnyxTokenizer(cfg.train.tokenizer_path)
    cfg.model.vocab_size = tokenizer.vocab_size
    model = OnyxLiteAI(cfg.model).to(device)
    print(f"onyxLiteAI | device={device} | params={model.parameter_count():,} | vocab={cfg.model.vocab_size}")

    meta_path = Path(cfg.train.data_dir) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"dtype": "uint16"}
    train_data = PackedTokenDataset(Path(cfg.train.data_dir) / "train.bin", cfg.model.block_size, device, meta.get("dtype", "uint16"))
    val_data = PackedTokenDataset(Path(cfg.train.data_dir) / "val.bin", cfg.model.block_size, device, meta.get("dtype", "uint16"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay, betas=(0.9, 0.95))
    start_step = 0
    best_val_loss = float("inf")

    if cfg.train.resume:
        checkpoint = torch.load(cfg.train.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        print(f"resumed from {cfg.train.resume} at step={start_step}")

    if cfg.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    amp_enabled = cfg.train.amp.lower() in {"fp16", "bf16"} and device in {"cuda", "cpu"}
    amp_dtype = torch.float16 if cfg.train.amp.lower() == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and cfg.train.amp.lower() == "fp16"))

    log_path = Path(cfg.train.out_dir) / "training_log.csv"
    new_log = not log_path.exists() or start_step == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr", "seconds"])
        if new_log:
            writer.writeheader()

        t0 = time.time()
        for step in range(start_step, cfg.train.max_steps + 1):
            lr = get_lr(step, cfg.train.max_steps, cfg.train.warmup_steps, cfg.train.learning_rate, cfg.train.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step % cfg.train.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data, cfg.train.eval_iters, cfg.train.batch_size)
                elapsed = time.time() - t0
                print(f"step={step:06d} train={losses['train']:.4f} val={losses['val']:.4f} lr={lr:.2e} sec={elapsed:.1f}")
                writer.writerow({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr, "seconds": elapsed})
                f.flush()
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    torch.save(build_checkpoint(model, optimizer, cfg.model, step, best_val_loss, {"tokenizer_path": cfg.train.tokenizer_path}), Path(cfg.train.out_dir) / "best.pt")

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for micro_step in range(cfg.train.grad_accum_steps):
                xb, yb = train_data.get_batch(cfg.train.batch_size)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    _, loss = model(xb, yb)
                    loss = loss / cfg.train.grad_accum_steps
                loss_accum += loss.item()
                scaler.scale(loss).backward()

            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if step > 0 and step % cfg.train.save_interval == 0:
                ckpt_path = Path(cfg.train.out_dir) / f"ckpt_step_{step}.pt"
                torch.save(build_checkpoint(model, optimizer, cfg.model, step, best_val_loss, {"tokenizer_path": cfg.train.tokenizer_path}), ckpt_path)

        final_path = Path(cfg.train.out_dir) / "final.pt"
        torch.save(build_checkpoint(model, optimizer, cfg.model, cfg.train.max_steps, best_val_loss, {"tokenizer_path": cfg.train.tokenizer_path}), final_path)
        print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
