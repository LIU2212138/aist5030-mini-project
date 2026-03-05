import os
import glob
import argparse

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def find_event_files(logdir: str):
    # 搜索 logdir 下所有 tfevents 文件
    patterns = [
        os.path.join(logdir, "**", "events.out.tfevents.*"),
        os.path.join(logdir, "events.out.tfevents.*"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # 去重+按时间排序（粗略）
    files = sorted(set(files), key=lambda x: os.path.getmtime(x))
    return files


def load_scalars(event_file: str):
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # load all
        },
    )
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    return ea, tags


def pick_loss_tag(tags):
    # 常见 loss tag 名字兼容：不同 transformers 版本可能写法不同
    candidates = [
        "train/loss",
        "loss",
        "training_loss",
        "train_loss",
    ]
    for c in candidates:
        if c in tags:
            return c
    # 兜底：找包含 loss 的 tag
    for t in tags:
        if "loss" in t.lower():
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="outputs/logs/oft_sst2")
    ap.add_argument("--out", type=str, default="figures/loss_curve.png")
    ap.add_argument("--title", type=str, default="Training Loss Curve")
    args = ap.parse_args()

    event_files = find_event_files(args.logdir)
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {args.logdir}")

    # 默认用最新的 event 文件
    event_file = event_files[-1]
    ea, tags = load_scalars(event_file)
    loss_tag = pick_loss_tag(tags)
    if loss_tag is None:
        raise RuntimeError(f"Cannot find a loss scalar tag in event file.\nAvailable scalar tags: {tags}")

    events = ea.Scalars(loss_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure()
    plt.plot(steps, values)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"[OK] Saved loss curve to: {args.out}")
    print(f"[INFO] Used event file: {event_file}")
    print(f"[INFO] Used loss tag: {loss_tag}")


if __name__ == "__main__":
    main()