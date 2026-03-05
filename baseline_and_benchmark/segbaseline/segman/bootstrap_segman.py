#!/usr/bin/env python3
"""
Bootstrap train–eval–test runner for SegMAN.

- Uses your train/val/test splitter (train_test_split.py) to re-split the full dataset per seed.
- For each seed, writes a per-seed train config (overriding train/val/test/out_dir/seed),
  then calls your trainer (train_segman.py) which trains, early-stops, and evaluates on test.
- Collects results into a JSON list, each item like:
  {
    "seed": 10,
    "overall_accuracy": 0.88256,
    "per_class_results": { ... }   # includes 'sky', 'cloud', 'contamination', 'accuracy', 'macro avg', 'weighted avg'
  }

Seeds are fixed to 20 values: 10, 27, 44, …, 333 (step = 17) unless overridden in config.
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
import subprocess


def arithmetic_seed_list(start: int = 10, end: int = 333, num: int = 20) -> list[int]:
    # 20 terms from 10 to 333 inclusive → step = (333-10)/(20-1) = 17
    step = (end - start) // (num - 1)
    return [start + step * i for i in range(num)]


def run(cmd: list[str], log_file: Path | None = None, cwd: Path | None = None) -> None:
    print(">>", " ".join(map(str, cmd)))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(cwd) if cwd else None,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    ret = proc.wait()
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("".join(lines), encoding="utf-8")
    if ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}): {' '.join(map(str, cmd))}")


def main():
    ap = argparse.ArgumentParser(description="Bootstrap SegMAN train–eval–test over multiple seeds.")
    ap.add_argument("--config", required=True, help="Path to bootstrap controller JSON.")
    ap.add_argument("--dry_run", action="store_true", help="Prepare splits/configs only; skip training.")
    args = ap.parse_args()

    ctrl = json.loads(Path(args.config).read_text(encoding="utf-8"))

    # Controller settings (with sensible defaults)
    data_dir = Path(ctrl["data_dir"]).expanduser()
    split_root = Path(ctrl.get("split_root", "./splits")).expanduser()
    out_root = Path(ctrl.get("out_root", "./runs/bootstrap")).expanduser()
    trainer_config_path = Path(ctrl["trainer_config"]).expanduser()

    split_script = Path(ctrl.get("split_script", "train_test_split.py")).expanduser()
    train_script = Path(ctrl.get("train_script", "train_segman.py")).expanduser()

    ratio_val = float(ctrl.get("ratio_val", 0.1))
    ratio_test = float(ctrl.get("ratio_test", 0.1))

    seeds = ctrl.get("seeds") or arithmetic_seed_list(10, 333, 20)  # fixed 20 seeds if none provided
    results_file = Path(ctrl.get("results_file", out_root / "bootstrap_results.json")).expanduser()

    base_train_cfg = json.loads(trainer_config_path.read_text(encoding="utf-8"))

    results: list[dict] = []

    for seed in seeds:
        tag = f"seed_{seed}"
        split_dir = split_root / tag
        out_dir = out_root / tag
        log_dir = out_dir

        # 1) Fresh split per seed
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

        split_cmd = [
            sys.executable, str(split_script),
            str(data_dir), str(split_dir),
            "--ratio_val", str(ratio_val),
            "--ratio_test", str(ratio_test),
            "--seed", str(seed),
            "--recursive",
            "--overwrite",
        ]
        run(split_cmd, log_file=log_dir / "split.log")

        # 2) Per-seed training config (override dirs/out_dir/seed)
        per_cfg = dict(base_train_cfg)
        per_cfg.update({
            "train_dir": str(split_dir / "train"),
            "val_dir": str(split_dir / "val"),
            "test_dir": str(split_dir / "test"),
            "out_dir": str(out_dir),
            "seed": int(seed),
        })
        per_cfg_path = out_dir / "config.json"
        per_cfg_path.parent.mkdir(parents=True, exist_ok=True)
        per_cfg_path.write_text(json.dumps(per_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        # 3) Train & eval (your trainer script handles training, early-stop, and test eval)
        if not args.dry_run:
            train_cmd = [sys.executable, str(train_script), "--config", str(per_cfg_path)]
            run(train_cmd, log_file=log_dir / "train.log")

            # 4) Collect the test metrics it wrote
            test_metrics_path = out_dir / "test_metrics.json"
            if not test_metrics_path.exists():
                raise FileNotFoundError(f"Expected test metrics at: {test_metrics_path}")
            rep = json.loads(test_metrics_path.read_text(encoding="utf-8"))

            # Normalize to the required shape
            per_class = rep.get("per_class_results", {})
            overall = rep.get("test_overall_accuracy", per_class.get("accuracy"))
            results.append({
                "seed": int(seed),
                "overall_accuracy": float(overall) if overall is not None else None,
                "per_class_results": per_class,
            })

            # Persist incrementally so you don't lose progress on long runs
            results_file.parent.mkdir(parents=True, exist_ok=True)
            results_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n✅ Bootstrap completed. Final results: {results_file}")


if __name__ == "__main__":
    main()