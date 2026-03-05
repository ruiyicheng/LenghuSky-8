import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================
# Configuration
# ============================================================
DATA_DIR = Path(r"D:\project\dino\cloud\data\segmentation\data")
SPLIT_ROOT = Path(r"D:\project\dino\cloud\data\segmentation")
TRAIN_DIR = SPLIT_ROOT / "train"
VAL_DIR   = SPLIT_ROOT / "val"
TEST_DIR  = SPLIT_ROOT / "test"

TRAIN_PY = Path(r"D:\project\dino\cloud\baseline\testcloud\train_segmentation.py")

# Base config for segmentation (now includes val root)
BASE_CONFIG = {
    "train_labelme_root": str(TRAIN_DIR),
    "val_labelme_root": str(VAL_DIR),     # <— added
    "test_labelme_root": str(TEST_DIR),
    "checkpoint_dir": r"D:\project\dino\cloud\log\cloudSegNet_feature_dino\checkpoint",
    "inference_out_dir": r"D:\project\dino\cloud\log\cloudSegNet_feature_dino\inference",
    "num_epochs": 500,
    "random_state": 42  # Will be overwritten per seed
}

# Where we write per-seed configs
BOOTSTRAP_CFG_DIR = Path(r"D:\project\dino\cloud\bootstrap_configs\segmentation")

# Logs & final aggregated CSV
LOG_DIR = Path(r"D:\project\dino\cloud\log\bootstrap_logs_segmentation")
METRICS_CSV = Path(r"D:\project\dino\cloud\log\bootstrap_metrics_segmentation.csv")

# ============================================================
# Split parameters (now train/val/test)
# ============================================================
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RECURSIVE = True

N_BOOTSTRAPS = 20
SEEDS = [10 + i * 17 for i in range(N_BOOTSTRAPS)]


def result_json_path(inference_out_dir: Path, seed: int) -> Path:
    # Keep your original convention for test results
    return Path(inference_out_dir) / f"test_results_seed_{seed}.json"


# ------------------ File Management ------------------

def gather_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.json") if p.is_file()])
    else:
        return sorted([p for p in root.glob("*.json") if p.is_file()])


def split_files(files: List[Path], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    """Shuffle deterministically and split into train/val/test."""
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("val_ratio must be in [0, 1).")
    if not (0.0 <= test_ratio < 1.0):
        raise ValueError("test_ratio must be in [0, 1).")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError(f"val_ratio + test_ratio must be < 1.0 (got {val_ratio + test_ratio})")

    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)

    n_val = int(math.floor(n * val_ratio))
    n_test = int(math.floor(n * test_ratio))
    n_train = n - n_val - n_test

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train:n_train + n_val]
    test_files = shuffled[n_train + n_val:]

    return train_files, val_files, test_files


def copy_preserving_structure(files: List[Path], base: Path, dest_root: Path) -> None:
    for src in files:
        rel = src.relative_to(base)
        dest_dir = (dest_root / rel.parent) if str(rel.parent) != "." else dest_root
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)


def regenerate_split(seed: int) -> None:
    """Overwrite train/, val/, and test/ directories with a new split."""
    # Clean existing directories
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if d.exists():
            shutil.rmtree(d)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Gather and split
    all_files = gather_files(DATA_DIR, RECURSIVE)
    if not all_files:
        raise RuntimeError(f"No .json files found in {DATA_DIR} {'(recursive)' if RECURSIVE else '(non-recursive)'}")

    train_files, val_files, test_files = split_files(all_files, VAL_RATIO, TEST_RATIO, seed)

    # Copy
    copy_preserving_structure(train_files, DATA_DIR, TRAIN_DIR)
    copy_preserving_structure(val_files, DATA_DIR, VAL_DIR)
    copy_preserving_structure(test_files, DATA_DIR, TEST_DIR)

    # Report
    n_all = len(all_files)
    print(
        f"Split regenerated (seed {seed}): "
        f"{len(train_files)} train ({len(train_files)/n_all:.1%}), "
        f"{len(val_files)} val ({len(val_files)/n_all:.1%}), "
        f"{len(test_files)} test ({len(test_files)/n_all:.1%})"
    )


# ------------------ Config Management ------------------

def seed_config(seed: int) -> Path:
    """Create a seed-specific config file (extends BASE_CONFIG)."""
    cfg = BASE_CONFIG.copy()
    cfg["random_state"] = int(seed)
    cfg["inference_out_dir"] = str(Path(cfg["inference_out_dir"]) / f"bootstrap_seed_{seed}")

    dst = BOOTSTRAP_CFG_DIR / f"segmentation_seed{seed}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)

    with open(dst, 'w', encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    return dst


# ------------------ Training & Metrics ------------------

def run_training(config_path: Path) -> str:
    """Launch training script and return stdout."""
    cmd = [sys.executable, str(TRAIN_PY), "--config", str(config_path)]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8"
    )
    return proc.stdout


def wait_for_file(p: Path, timeout_sec: float = 30.0, poll_sec: float = 0.2) -> bool:
    waited = 0.0
    while waited < timeout_sec:
        if p.exists():
            return True
        time.sleep(poll_sec)
        waited += poll_sec
    return p.exists()


def extract_metrics_from_result_json(j: Dict) -> Dict[str, float]:
    """Extract flat metrics from result JSON (same schema as before)."""
    out = {k: "" for k in [
        "accuracy",
        "sky_precision", "sky_recall", "sky_f1",
        "cloud_precision", "cloud_recall", "cloud_f1",
        "contamination_precision", "contamination_recall", "contamination_f1",
    ]}

    # Overall accuracy
    acc = j.get("overall_accuracy", None)
    if acc is None:
        acc = j.get("per_class_results", {}).get("accuracy", None)
    if acc is not None:
        out["accuracy"] = float(acc)

    # Per-class metrics
    pcr = j.get("per_class_results", {})

    def put(cls_name: str, dest_prefix: str):
        if cls_name in pcr:
            d = pcr[cls_name]
            if "precision" in d:
                out[f"{dest_prefix}_precision"] = float(d["precision"])
            if "recall" in d:
                out[f"{dest_prefix}_recall"] = float(d["recall"])
            if "f1-score" in d:
                out[f"{dest_prefix}_f1"] = float(d["f1-score"])

    put("sky", "sky")
    put("cloud", "cloud")
    put("contamination", "contamination")

    return out


# ------------------ Main ------------------

def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_CFG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ensure_dirs()

    # Validate ratios once (static across seeds)
    if VAL_RATIO + TEST_RATIO >= 1.0:
        raise RuntimeError(
            f"VAL_RATIO + TEST_RATIO must be < 1.0 (got {VAL_RATIO + TEST_RATIO})"
        )

    # Prepare CSV
    write_header = not METRICS_CSV.exists()
    with METRICS_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow([
                "seed", "config",
                "accuracy",
                "sky_precision", "sky_recall", "sky_f1",
                "cloud_precision", "cloud_recall", "cloud_f1",
                "contamination_precision", "contamination_recall", "contamination_f1",
            ])

        for seed in SEEDS:
            print(f"\n=== Bootstrap seed={seed} ===")

            try:
                # Regenerate train/val/test split
                regenerate_split(seed)

                # Create seed-specific config
                seeded_cfg = seed_config(seed)
                print(f"Running training with {seeded_cfg.name} ...")

                # Run training
                stdout_text = run_training(seeded_cfg)

                # Save log
                (LOG_DIR / f"segmentation_seed{seed}.log").write_text(stdout_text, encoding="utf-8")

                # Find result JSON
                with open(seeded_cfg, 'r', encoding="utf-8") as f:
                    cfg_data = json.load(f)
                inf_out_dir = Path(cfg_data["inference_out_dir"])
                res_json = result_json_path(inf_out_dir, seed)

                # Wait for results
                if not wait_for_file(res_json):
                    print(f"seed={seed} FAILED: {res_json} not found.")
                    writer.writerow([seed, "segmentation"] + [""] * 10)
                    csvfile.flush()
                    continue

                # Parse metrics
                try:
                    with open(res_json, 'r', encoding="utf-8") as f:
                        result_obj = json.load(f)
                    flat = extract_metrics_from_result_json(result_obj)
                except Exception as e:
                    print(f"seed={seed} FAILED to parse JSON: {e}")
                    writer.writerow([seed, "segmentation"] + [""] * 10)
                    csvfile.flush()
                    continue

                # Write results
                row = [
                    seed, "segmentation",
                    flat["accuracy"],
                    flat["sky_precision"], flat["sky_recall"], flat["sky_f1"],
                    flat["cloud_precision"], flat["cloud_recall"], flat["cloud_f1"],
                    flat["contamination_precision"], flat["contamination_recall"], flat["contamination_f1"],
                ]
                writer.writerow(row)
                csvfile.flush()

                acc_disp = flat["accuracy"] if flat["accuracy"] != "" else "NA"
                print(f"seed={seed} accuracy={acc_disp} -> {res_json}")

            except Exception as e:
                print(f"seed={seed} FAILED with exception: {e}")
                writer.writerow([seed, "segmentation"] + [""] * 10)
                csvfile.flush()
                continue

    print(f"\nDone. Aggregated CSV -> {METRICS_CSV}")
    print(f"Logs -> {LOG_DIR}")


if __name__ == "__main__":
    main()
