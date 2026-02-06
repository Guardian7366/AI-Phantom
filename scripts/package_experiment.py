import os
import json
import shutil
import hashlib
import time
from pathlib import Path


# -------------------------------------------------
# Utils
# -------------------------------------------------

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def safe_copy(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("=== Packaging Final Experiment ===")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    package_dir = Path("results") / f"experiment_{timestamp}"
    package_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Paths
    # ----------------------------

    results_dir = Path("results")
    scripts_dir = Path("scripts")
    configs_dir = Path("configs")

    best_model_dir = results_dir / "best_model"
    plots_dir = results_dir / "plots"

    # ----------------------------
    # Manifest
    # ----------------------------

    manifest = {
        "timestamp": timestamp,
        "files": {},
        "hashes": {},
    }

    # ----------------------------
    # Copy configs
    # ----------------------------

    cfg_dst = package_dir / "configs"
    cfg_dst.mkdir(exist_ok=True)

    for cfg in configs_dir.glob("*.yaml"):
        dst = cfg_dst / cfg.name
        safe_copy(cfg, dst)
        manifest["files"][str(dst)] = str(cfg)
        manifest["hashes"][cfg.name] = hash_file(cfg)

    # ----------------------------
    # Copy best model
    # ----------------------------

    model_dst = package_dir / "best_model"
    model_dst.mkdir(exist_ok=True)

    for file in ["best_model.pth", "metadata.json"]:
        src = best_model_dir / file
        dst = model_dst / file
        safe_copy(src, dst)
        if src.exists():
            manifest["hashes"][file] = hash_file(src)

    # ----------------------------
    # Copy evaluation artifacts
    # ----------------------------

    eval_dst = package_dir / "evaluation"
    eval_dst.mkdir(exist_ok=True)

    for file in plots_dir.glob("*.json"):
        dst = eval_dst / file.name
        safe_copy(file, dst)
        manifest["hashes"][file.name] = hash_file(file)

    # ----------------------------
    # Save manifest
    # ----------------------------

    manifest_path = package_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Experiment packaged at: {package_dir}")
    print("[OK] Manifest created")


if __name__ == "__main__":
    main()
