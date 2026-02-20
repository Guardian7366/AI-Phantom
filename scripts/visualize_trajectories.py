import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=str, required=True, help="results/trajectories/trajectories.json")
    args = ap.parse_args()

    with open(args.traj, "r", encoding="utf-8") as f:
        payload = json.load(f)

    trajs = payload.get("trajectories", [])
    lengths = []
    success = 0

    for t in trajs:
        steps = t.get("steps", [])
        lengths.append(len(steps))
        if steps and steps[-1].get("done", False):
            success += 1

    print("Trajectories:", len(trajs))
    if trajs:
        print("Success rate:", success / len(trajs))
        print("Mean length:", float(np.mean(lengths)))
        print("Min/Max length:", int(np.min(lengths)), int(np.max(lengths)))


if __name__ == "__main__":
    main()
