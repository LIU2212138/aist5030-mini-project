import os
import json
import argparse
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", type=str, default="outputs/checkpoints/oft_sst2/trainer_state.json")
    ap.add_argument("--out", type=str, default="figures/loss_curve.png")
    ap.add_argument("--title", type=str, default="Training Loss Curve")
    args = ap.parse_args()

    if not os.path.exists(args.state):
        raise FileNotFoundError(f"trainer_state.json not found: {args.state}")

    with open(args.state, "r", encoding="utf-8") as f:
        st = json.load(f)

    # transformers 会把日志放在 log_history 里
    hist = st.get("log_history", [])
    steps = []
    losses = []
    for item in hist:
        # 只取训练 loss（不是 eval_loss）
        if "loss" in item and "step" in item and "eval_loss" not in item:
            steps.append(item["step"])
            losses.append(item["loss"])

    if not steps:
        raise RuntimeError("No training loss found in log_history. "
                           "Try increasing logging_steps or check other log files.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] Saved loss curve to: {args.out}")
    print(f"[INFO] Points: {len(steps)}")


if __name__ == "__main__":
    main()