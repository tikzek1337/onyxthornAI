from __future__ import annotations

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot onyxthornAI chat training losses")
    parser.add_argument("--log", default="runs/onyxliteai_chat_tiny/training_log.csv")
    parser.add_argument("--out", default="runs/onyxliteai_chat_tiny/loss_curve.png")
    args = parser.parse_args()

    df = pd.read_csv(args.log)
    plt.figure()
    plt.plot(df["step"], df["train_loss"], label="train")
    plt.plot(df["step"], df["val_loss"], label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("onyxthornAI chat training loss")
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
