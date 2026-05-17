from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from .config import load_project_config
from .data import PackedTokenDataset
from .model import OnyxThornAI, build_checkpoint
from .tokenizer import OnyxTokenizer


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")


def get_lr(step: int, max_steps: int, warmup_steps: int, learning_rate: float, min_lr: float) -> float:
    if step < warmup_steps:
        return learning_rate * step / max(1, warmup_steps)
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def estimate_model_storage_mb(param_count: int, bytes_per_param: int = 2) -> float:
    return param_count * bytes_per_param / (1024 ** 2)


def make_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, device: str) -> torch.optim.Optimizer:
    kwargs: dict[str, Any] = dict(lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    if device == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except TypeError:
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def make_inference_checkpoint(checkpoint: dict, dtype: str = "fp16") -> dict:
    if dtype not in {"fp32", "fp16", "bf16"}:
        raise ValueError("slim dtype must be one of: fp32, fp16, bf16")
    out = dict(checkpoint)
    out["optimizer_state"] = None
    converted = {}
    for key, tensor in checkpoint["model_state"].items():
        t = tensor.detach().cpu()
        if dtype == "fp16" and torch.is_floating_point(t):
            t = t.half()
        elif dtype == "bf16" and torch.is_floating_point(t):
            t = t.bfloat16()
        elif dtype == "fp32" and torch.is_floating_point(t):
            t = t.float()
        converted[key] = t
    out["model_state"] = converted
    out.setdefault("extra", {})["inference_only"] = True
    out.setdefault("extra", {})["weight_dtype"] = dtype
    return out


@torch.no_grad()
def estimate_loss(model: OnyxThornAI, train_data: PackedTokenDataset, val_data: PackedTokenDataset, eval_iters: int, batch_size: int) -> dict:
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
    parser = argparse.ArgumentParser(description="Train onyxthornAI chat model")
    parser.add_argument("--config", type=str, default="configs/onyxthornai_chat_17m.yaml")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt")
    parser.add_argument("--max_steps", type=int, default=None, help="Override train.max_steps from YAML")
    parser.add_argument("--batch_size", type=int, default=None, help="Override train.batch_size from YAML")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Override train.grad_accum_steps from YAML")
    parser.add_argument("--save_slim_final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slim_dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    if args.resume:
        cfg.train.resume = args.resume
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max_steps must be a positive integer")
        cfg.train.max_steps = args.max_steps
        cfg.train.warmup_steps = min(cfg.train.warmup_steps, max(1, cfg.train.max_steps // 10))
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.train.grad_accum_steps = args.grad_accum_steps

    torch.manual_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)
    Path(cfg.train.out_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = OnyxTokenizer(cfg.train.tokenizer_path)
    cfg.model.vocab_size = tokenizer.vocab_size
    model = OnyxThornAI(cfg.model).to(device)
    param_count = model.parameter_count()
    print(
        f"onyxthornAI | device={device} | params={param_count:,} | vocab={cfg.model.vocab_size} | "
        f"context={cfg.model.block_size} tokens | fp16 size≈{estimate_model_storage_mb(param_count, 2):.1f} MB | fp32 size≈{estimate_model_storage_mb(param_count, 4):.1f} MB"
    )
    print(
        f"training | steps={cfg.train.max_steps:,} | batch={cfg.train.batch_size} | accum={cfg.train.grad_accum_steps} | "
        f"tokens/update≈{cfg.train.batch_size * cfg.train.grad_accum_steps * cfg.model.block_size:,}"
    )

    meta_path = Path(cfg.train.data_dir) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"dtype": "uint16"}
    train_data = PackedTokenDataset(Path(cfg.train.data_dir) / "train.bin", cfg.model.block_size, device, meta.get("dtype", "uint16"))
    val_data = PackedTokenDataset(Path(cfg.train.data_dir) / "val.bin", cfg.model.block_size, device, meta.get("dtype", "uint16"))

    optimizer = make_optimizer(model, cfg.train.learning_rate, cfg.train.weight_decay, device)
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
        writer = csv.DictWriter(f, fieldnames=["step", "train_loss", "val_loss", "lr", "seconds", "eta_seconds"])
        if new_log:
            writer.writeheader()

        t0 = time.time()
        last_eval_time = t0
        last_eval_step = start_step
        for step in range(start_step + 1, cfg.train.max_steps + 1):
            lr = get_lr(step, cfg.train.max_steps, cfg.train.warmup_steps, cfg.train.learning_rate, cfg.train.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            if step % cfg.train.eval_interval == 0:
                losses = estimate_loss(model, train_data, val_data, cfg.train.eval_iters, cfg.train.batch_size)
                elapsed = time.time() - t0
                now = time.time()
                steps_since_eval = max(1, step - last_eval_step)
                sec_per_step = (now - last_eval_time) / steps_since_eval if step != last_eval_step else 0.0
                eta = max(0, cfg.train.max_steps - step) * sec_per_step
                last_eval_time = now
                last_eval_step = step
                print(
                    f"step={step:06d} train={losses['train']:.4f} val={losses['val']:.4f} lr={lr:.2e} "
                    f"sec={elapsed:.1f} eta≈{eta:.0f}s"
                )
                writer.writerow({"step": step, "train_loss": losses["train"], "val_loss": losses["val"], "lr": lr, "seconds": elapsed, "eta_seconds": eta})
                f.flush()
                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    ckpt = build_checkpoint(model, optimizer, cfg.model, step, best_val_loss, {"tokenizer_path": cfg.train.tokenizer_path})
                    torch.save(ckpt, Path(cfg.train.out_dir) / "best.pt")
                    if args.save_slim_final:
                        torch.save(make_inference_checkpoint(ckpt, args.slim_dtype), Path(cfg.train.out_dir) / f"best_infer_{args.slim_dtype}.pt")

            optimizer.zero_grad(set_to_none=True)
            for _ in range(cfg.train.grad_accum_steps):
                xb, yb = train_data.get_batch(cfg.train.batch_size)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_enabled):
                    _, loss = model(xb, yb)
                    loss = loss / cfg.train.grad_accum_steps
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
        final_ckpt = build_checkpoint(model, optimizer, cfg.model, cfg.train.max_steps, best_val_loss, {"tokenizer_path": cfg.train.tokenizer_path})
        torch.save(final_ckpt, final_path)
        print(f"saved final checkpoint: {final_path}")
        if args.save_slim_final:
            slim_path = Path(cfg.train.out_dir) / f"final_infer_{args.slim_dtype}.pt"
            torch.save(make_inference_checkpoint(final_ckpt, args.slim_dtype), slim_path)
            print(f"saved slim inference checkpoint: {slim_path}")


if __name__ == "__main__":
    main()
