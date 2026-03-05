import os
import argparse
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=float, default=0.9113)
    ap.add_argument("--after", type=float, default=0.9525)
    ap.add_argument("--out", type=str, default="figures/acc_comparison.png")
    ap.add_argument("--title", type=str, default="Accuracy Before vs After OFT Finetuning")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    labels = ["Before", "After (OFT)"]
    values = [args.before, args.after]

    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(args.title)

    # 在柱子顶上标注数值
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] Saved accuracy comparison to: {args.out}")


if __name__ == "__main__":
    main()