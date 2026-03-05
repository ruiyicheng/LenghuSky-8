# extract_dinov3_global.py
# -*- coding: utf-8 -*-
"""
Using DINOv3 to extract global features (CLS token) for two-class background (l/u) images,

Output:
  X_train.npy, y_train.npy, paths_train.txt
  X_test.npy,  y_test.npy,  paths_test.txt
  meta.json  


python binary_2023_two_class_tokenization.py --dir_l "D:\\project\\dino\\cloud\\data\\bkg_mask\\train_new_facility_img\\l" --dir_u "D:\\project\\dino\\cloud\\data\\bkg_mask\\train_new_facility_img\\u"   --out   "D:\\project\\dino\\cloud\\data\\bkg_mask\\tokens"   --batch-size 16 --seed 42
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


def set_seed(seed: int = 42):
    """固定所有可控随机性，并启用确定性计算。"""
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确定性（在有些 CUDA/驱动上会略降速，但更可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_images(folder: str) -> List[str]:
    """按文件名排序收集图片路径（jpg/jpeg/png）。"""
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [str(p) for p in Path(folder).glob("*") if p.suffix.lower() in exts]
    paths.sort()  # 关键：文件名顺序
    return paths


def make_split_per_class(
    paths_l: List[str], paths_u: List[str], ratio: float = 0.9
) -> Tuple[List[str], List[int], List[str], List[int]]:
    """对每个类别内部按顺序切分前 90% 为训练，其余为测试。"""
    def split_one(paths):
        n = len(paths)
        n_tr = int(math.floor(n * ratio))
        return paths[:n_tr], paths[n_tr:]

    tr_l, te_l = split_one(paths_l)
    tr_u, te_u = split_one(paths_u)

    train_paths = tr_l + tr_u
    train_labels = [0] * len(tr_l) + [1] * len(tr_u)
    test_paths = te_l + te_u
    test_labels = [0] * len(te_l) + [1] * len(te_u)

    # 为了后续读取/索引稳定，再各自按路径排序一次（可省略）
    train = sorted(zip(train_paths, train_labels), key=lambda x: x[0])
    test = sorted(zip(test_paths, test_labels), key=lambda x: x[0])

    train_paths, train_labels = [p for p, _ in train], [y for _, y in train]
    test_paths, test_labels = [p for p, _ in test], [y for _, y in test]

    return train_paths, train_labels, test_paths, test_labels


@torch.inference_mode()
def extract_features(
    paths: List[str],
    processor: AutoImageProcessor,
    model: AutoModel,
    batch_size: int = 16,
    target_size: int = 512,
    device: str = "cuda",
    use_cls: bool = True,
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    批量抽取全局特征。
    - use_cls=True 取 CLS token 作为 global feature（推荐用于 DINOv3）
    - 若改为 use_cls=False，可改为 patch mean pooling
    """
    feats = []

    # 为了尽量稳定，明确 resize 大小，且禁用 center crop
    def pil_loader(p):
        return Image.open(p).convert("RGB")

    num_register = getattr(model.config, "num_register_tokens", 0)
    patch = getattr(model.config, "patch_size", 16)

    pbar = tqdm(range(0, len(paths), batch_size), desc="Extracting", ncols=88)
    for i in pbar:
        batch_paths = paths[i : i + batch_size]
        imgs = [pil_loader(p) for p in batch_paths]
        inputs = processor(
            images=imgs,
            size={"height": target_size, "width": target_size},
            do_center_crop=False,
            return_tensors="pt",
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [B, 1 + reg + tokens, C]
        if use_cls:
            feat = hidden[:, 0, :]  # CLS
        else:
            # 平均所有 patch token（跳过 CLS 与 register tokens）
            feat = hidden[:, 1 + num_register :, :].mean(dim=1)

        if l2_normalize:
            feat = F.normalize(feat, dim=1)
        feats.append(feat.cpu().numpy())

        pbar.set_postfix(batch=f"{i//batch_size+1}/{math.ceil(len(paths)/batch_size)}")

    feats = np.concatenate(feats, axis=0)
    return feats


def save_split(
    out_dir: str,
    X_tr: np.ndarray,
    y_tr: List[int],
    paths_tr: List[str],
    X_te: np.ndarray,
    y_te: List[int],
    paths_te: List[str],
    meta: dict,
):
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_tr)
    np.save(os.path.join(out_dir, "y_train.npy"), np.array(y_tr, dtype=np.int64))
    np.save(os.path.join(out_dir, "X_test.npy"),  X_te)
    np.save(os.path.join(out_dir, "y_test.npy"),  np.array(y_te, dtype=np.int64))

    with open(os.path.join(out_dir, "paths_train.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(paths_tr))
    with open(os.path.join(out_dir, "paths_test.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(paths_te))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {out_dir}")
    for fn in ["X_train.npy","y_train.npy","X_test.npy","y_test.npy","paths_train.txt","paths_test.txt","meta.json"]:
        print(" -", fn)


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv3 global features (CLS) and save as npy.")
    parser.add_argument("--dir_l", required=True, help="类别0（l）图片文件夹")
    parser.add_argument("--dir_u", required=True, help="类别1（u）图片文件夹")
    parser.add_argument("--out",   required=True, help="输出目录，保存 .npy 与元数据")
    parser.add_argument("--model-id", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="HuggingFace 模型名")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20250101)
    parser.add_argument("--target-size", type=int, default=512,
                        help="明确的resize大小（确保被patch size整除；默认512）")
    parser.add_argument("--no-cuda", action="store_true", help="仅用CPU推理")
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls",
                        help="global feature 的方式：cls=CLS token；mean=patch mean")
    parser.add_argument("--no-l2", action="store_true", help="不做L2归一化（默认会归一化）")
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    # 1) 收集与切分
    paths_l = list_images(args.dir_l)
    paths_u = list_images(args.dir_u)
    assert len(paths_l) > 0 and len(paths_u) > 0, "两类文件夹里都必须有图片！"
    print(f"Found L: {len(paths_l)} images; U: {len(paths_u)} images.")

    tr_paths, tr_labels, te_paths, te_labels = make_split_per_class(paths_l, paths_u, ratio=0.9)
    print(f"Train: {len(tr_paths)}  Test: {len(te_paths)}  (per-class 9:1 by filename order)")

    # 2) 模型与处理器
    print("Loading model & processor...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.eval().to(device)

    # 3) 抽取特征
    use_cls = (args.pool == "cls")
    l2_norm = not args.no_l2

    X_tr = extract_features(
        tr_paths, processor, model,
        batch_size=args.batch_size, target_size=args.target_size,
        device=device, use_cls=use_cls, l2_normalize=l2_norm
    )
    X_te = extract_features(
        te_paths, processor, model,
        batch_size=args.batch_size, target_size=args.target_size,
        device=device, use_cls=use_cls, l2_normalize=l2_norm
    )

    # 4) 保存
    meta = {
        "model_id": args.model_id,
        "device": device,
        "feature": "CLS" if use_cls else "PATCH_MEAN",
        "l2_normalize": l2_norm,
        "target_size": args.target_size,
        "patch_size": getattr(model.config, "patch_size", None),
        "num_register_tokens": getattr(model.config, "num_register_tokens", 0),
        "class_map": {"l": 0, "u": 1},
        "counts": {
            "l_total": len(paths_l), "u_total": len(paths_u),
            "train": len(tr_paths), "test": len(te_paths)
        },
        "seed": args.seed,
        "split": "per-class filename-order 9:1",
        "paths": {"dir_l": os.path.abspath(args.dir_l), "dir_u": os.path.abspath(args.dir_u)},
    }

    save_split(args.out, X_tr, tr_labels, tr_paths, X_te, te_labels, te_paths, meta)


if __name__ == "__main__":
    main()
