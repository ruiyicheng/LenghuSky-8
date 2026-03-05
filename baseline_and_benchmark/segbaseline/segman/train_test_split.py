# #!/usr/bin/env python3
# """
# Split a folder of JSON files into train/test in a 7:3 proportion.

# Usage:
#   python train_test_split.py /path/to/data /path/to/output
#   # optional flags:
#   #   --ratio 0.7               Train proportion (default: 0.7)
#   #   --seed 42                 RNG seed for reproducibility (default: 42)
#   #   --recursive               Recurse into subdirectories (default: off)
#   #   --overwrite               Replace existing output/train & output/test
# """

# from __future__ import annotations
# import argparse
# import math
# import random
# import shutil
# from pathlib import Path
# import sys

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Split JSON files into train/test sets.")
#     p.add_argument("data_dir", type=Path, help="Folder containing .json files.")
#     p.add_argument("out_dir", type=Path, help="Folder where train/ and test/ will be created.")
#     p.add_argument("--ratio_test", type=float, default=0.1, help="Train proportion in [0,1]. Default: 0.9")
#     p.add_argument("--ratio_val", type=float, default=0.1, help="Train proportion in [0,1]. Default: 0.9")
#     p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
#     p.add_argument("--recursive", action="store_true", help="Include JSON files in subdirectories.")
#     p.add_argument("--overwrite", action="store_true", help="Allow replacing existing split folders.")
#     return p.parse_args()

# def validate_args(args: argparse.Namespace) -> None:
#     if not args.data_dir.exists() or not args.data_dir.is_dir():
#         sys.exit(f"ERROR: data_dir does not exist or is not a directory: {args.data_dir}")
#     if not (0.0 < args.ratio < 1.0):
#         sys.exit("ERROR: --ratio must be between 0 and 1 (exclusive).")
#     # Prepare output dirs
#     train_dir = args.out_dir / "train"
#     test_dir = args.out_dir / "test"
#     for d in (train_dir, test_dir):
#         if d.exists():
#             if not args.overwrite:
#                 sys.exit(f"ERROR: Output subdir exists: {d}\n"
#                          f"       Use --overwrite to replace it.")
#             # remove and recreate
#             shutil.rmtree(d)
#     train_dir.mkdir(parents=True, exist_ok=True)
#     test_dir.mkdir(parents=True, exist_ok=True)

# def gather_files(root: Path, recursive: bool) -> list[Path]:
#     if recursive:
#         return sorted([p for p in root.rglob("*.json") if p.is_file()])
#     else:
#         return sorted([p for p in root.glob("*.json") if p.is_file()])

# def split_files(files: list[Path], train_ratio: float, seed: int) -> tuple[list[Path], list[Path]]:
#     rng = random.Random(seed)
#     shuffled = files.copy()
#     rng.shuffle(shuffled)
#     n = len(shuffled)
#     n_train = int(math.floor(n * train_ratio))
#     return shuffled[:n_train], shuffled[n_train:]

# def copy_preserving_structure(files: list[Path], base: Path, dest_root: Path) -> None:
#     for src in files:
#         rel = src.relative_to(base)
#         dest_dir = (dest_root / rel.parent) if rel.parent != Path(".") else dest_root
#         dest_dir.mkdir(parents=True, exist_ok=True)
#         shutil.copy2(src, dest_dir / src.name)

# def write_manifest(files: list[Path], base: Path, out_path: Path) -> None:
#     rel_lines = [str(p.relative_to(base)).replace("\\", "/") for p in files]
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text("\n".join(rel_lines), encoding="utf-8")

# def main():
#     args = parse_args()
#     validate_args(args)

#     all_files = gather_files(args.data_dir, args.recursive)
#     if not all_files:
#         sys.exit(f"ERROR: No .json files found in {args.data_dir} "
#                  f"{'(recursive)' if args.recursive else '(non-recursive)'}.")

#     train_files, test_files = split_files(all_files, args.ratio, args.seed)

#     copy_preserving_structure(train_files, args.data_dir, args.out_dir / "train")
#     copy_preserving_structure(test_files, args.data_dir, args.out_dir / "test")

#     # Write manifests for convenience
#     write_manifest(train_files, args.data_dir, args.out_dir / "train.txt")
#     write_manifest(test_files, args.data_dir, args.out_dir / "test.txt")

#     print(f"Done.")
#     print(f"Total JSON files: {len(all_files)}")
#     print(f"Train: {len(train_files)}  ({len(train_files)/len(all_files):.1%})  -> {args.out_dir / 'train'}")
#     print(f"Test : {len(test_files)}  ({len(test_files)/len(all_files):.1%})  -> {args.out_dir / 'test'}")
#     print(f"Manifests written to: {args.out_dir / 'train.txt'} and {args.out_dir / 'test.txt'}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Split a folder of JSON files into train/val/test sets.

Usage:
  python train_test_split.py /path/to/data /path/to/output
  # optional flags:
  #   --ratio_val 0.1          Validation proportion (default: 0.1)
  #   --ratio_test 0.1         Test proportion (default: 0.1)
  #   --seed 42                RNG seed for reproducibility (default: 42)
  #   --recursive              Recurse into subdirectories (default: off)
  #   --overwrite              Replace existing output/train & output/val & output/test
"""

from __future__ import annotations
import argparse
import math
import random
import shutil
from pathlib import Path
import sys

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split JSON files into train/val/test sets.")
    p.add_argument("data_dir", type=Path, help="Folder containing .json files.")
    p.add_argument("out_dir", type=Path, help="Folder where train/, val/, and test/ will be created.")
    p.add_argument("--ratio_val", type=float, default=0.1, help="Validation proportion in [0,1]. Default: 0.1")
    p.add_argument("--ratio_test", type=float, default=0.1, help="Test proportion in [0,1]. Default: 0.1")
    p.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--recursive", action="store_true", help="Include JSON files in subdirectories.")
    p.add_argument("--overwrite", action="store_true", help="Allow replacing existing split folders.")
    return p.parse_args()

def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.exists() or not args.data_dir.is_dir():
        sys.exit(f"ERROR: data_dir does not exist or is not a directory: {args.data_dir}")
    
    if not (0.0 <= args.ratio_val < 1.0):
        sys.exit("ERROR: --ratio_val must be between 0 and 1 (exclusive).")
    if not (0.0 <= args.ratio_test < 1.0):
        sys.exit("ERROR: --ratio_test must be between 0 and 1 (exclusive).")
    
    total_ratio = args.ratio_val + args.ratio_test
    if total_ratio >= 1.0:
        sys.exit(f"ERROR: Sum of ratio_val ({args.ratio_val}) and ratio_test ({args.ratio_test}) must be less than 1.0 (current: {total_ratio})")
    
    # Prepare output dirs
    train_dir = args.out_dir / "train"
    val_dir = args.out_dir / "val"
    test_dir = args.out_dir / "test"
    
    for d in (train_dir, val_dir, test_dir):
        if d.exists():
            if not args.overwrite:
                sys.exit(f"ERROR: Output subdir exists: {d}\n"
                         f"       Use --overwrite to replace it.")
            # remove and recreate
            shutil.rmtree(d)
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

def gather_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.json") if p.is_file()])
    else:
        return sorted([p for p in root.glob("*.json") if p.is_file()])

def split_files(files: list[Path], val_ratio: float, test_ratio: float, seed: int) -> tuple[list[Path], list[Path], list[Path]]:
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

def copy_preserving_structure(files: list[Path], base: Path, dest_root: Path) -> None:
    for src in files:
        rel = src.relative_to(base)
        dest_dir = (dest_root / rel.parent) if rel.parent != Path(".") else dest_root
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / src.name)

def write_manifest(files: list[Path], base: Path, out_path: Path) -> None:
    rel_lines = [str(p.relative_to(base)).replace("\\", "/") for p in files]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rel_lines), encoding="utf-8")

def main():
    args = parse_args()
    validate_args(args)

    all_files = gather_files(args.data_dir, args.recursive)
    if not all_files:
        sys.exit(f"ERROR: No .json files found in {args.data_dir} "
                 f"{'(recursive)' if args.recursive else '(non-recursive)'}.")

    train_files, val_files, test_files = split_files(all_files, args.ratio_val, args.ratio_test, args.seed)

    # Copy files to respective directories
    copy_preserving_structure(train_files, args.data_dir, args.out_dir / "train")
    copy_preserving_structure(val_files, args.data_dir, args.out_dir / "val")
    copy_preserving_structure(test_files, args.data_dir, args.out_dir / "test")

    # Write manifests for convenience
    write_manifest(train_files, args.data_dir, args.out_dir / "train.txt")
    write_manifest(val_files, args.data_dir, args.out_dir / "val.txt")
    write_manifest(test_files, args.data_dir, args.out_dir / "test.txt")

    train_ratio = len(train_files) / len(all_files)
    val_ratio = len(val_files) / len(all_files)
    test_ratio = len(test_files) / len(all_files)
    
    print(f"Done.")
    print(f"Total JSON files: {len(all_files)}")
    print(f"Train: {len(train_files)}  ({train_ratio:.1%})  -> {args.out_dir / 'train'}")
    print(f"Val:   {len(val_files)}  ({val_ratio:.1%})  -> {args.out_dir / 'val'}")
    print(f"Test:  {len(test_files)}  ({test_ratio:.1%})  -> {args.out_dir / 'test'}")
    print(f"Manifests written to: {args.out_dir / 'train.txt'}, {args.out_dir / 'val.txt'}, {args.out_dir / 'test.txt'}")

if __name__ == "__main__":
    main()