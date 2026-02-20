import argparse
import json
import os
import matplotlib.pyplot as plt


def load_history(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=str, required=True,
                    help="Ruta a results/runs/<run_id>_history.json")
    ap.add_argument("--out_dir", type=str, default="results/plots")
    args = ap.parse_args()

    hist = load_history(args.history)
    os.makedirs(args.out_dir, exist_ok=True)

    ep = hist["episode"]
    reward = hist["reward"]
    success = hist["success"]
    epsilon = hist["epsilon"]
    loss = hist["loss"]

    # 1) Reward
    plt.figure()
    plt.plot(ep, reward)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    out1 = os.path.join(args.out_dir, "reward.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    # 2) Success (moving average simple)
    window = 200
    sm = []
    for i in range(len(success)):
        j0 = max(0, i - window + 1)
        sm.append(sum(success[j0:i+1]) / float(i - j0 + 1))

    plt.figure()
    plt.plot(ep, sm)
    plt.title(f"Success Rate (moving avg, window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate")
    out2 = os.path.join(args.out_dir, "success.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    # 3) Epsilon
    plt.figure()
    plt.plot(ep, epsilon)
    plt.title("Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    out3 = os.path.join(args.out_dir, "epsilon.png")
    plt.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close()

    # 4) Loss (ignorar None)
    xs = []
    ys = []
    for e, l in zip(ep, loss):
        if l is not None:
            xs.append(e)
            ys.append(l)

    if len(xs) > 10:
        plt.figure()
        plt.plot(xs, ys)
        plt.title("Loss (Huber)")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        out4 = os.path.join(args.out_dir, "loss.png")
        plt.savefig(out4, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        out4 = None

    # 5) Eval success over time
    eval_points = hist.get("eval", [])
    if eval_points:
        x = [p["episode"] for p in eval_points]
        y = [p["success_rate"] for p in eval_points]
        plt.figure()
        plt.plot(x, y)
        plt.title("Evaluation Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        out5 = os.path.join(args.out_dir, "eval_success.png")
        plt.savefig(out5, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        out5 = None

    print("Saved plots:")
    print(" -", out1)
    print(" -", out2)
    print(" -", out3)
    if out4:
        print(" -", out4)
    if out5:
        print(" -", out5)


if __name__ == "__main__":
    main()
