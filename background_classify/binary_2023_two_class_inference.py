# infer_dinov3_linear.py
# -*- coding: utf-8 -*-
"""
This script performs inference using a DINOv3 backbone combined with a linear classifier for binary classification.
python binary_2023_two_class_inference.py  --input-dir "D:\\project\\dino\\cloud\\data\\2023"  --model-pt  "D:\\project\\dino\\cloud\\data\\bkg_mask\\lincls\\model.pt"  --out-csv   "D:\\project\\dino\\cloud\\data\\bkg_mask\\inference_result\\2023.csv"  --batch-size 32 --seed 42
python binary_2023_two_class_inference.py  --input-dir "D:\\project\\dino\\cloud\\data\\2024"  --model-pt  "D:\\project\\dino\\cloud\\data\\bkg_mask\\lincls\\model.pt"  --out-csv   "D:\\project\\dino\\cloud\\data\\bkg_mask\\inference_result\\2024.csv"  --batch-size 32 --seed 42
python binary_2023_two_class_inference.py  --input-dir "D:\\project\\dino\\cloud\\data\\2025"  --model-pt  "D:\\project\\dino\\cloud\\data\\bkg_mask\\lincls\\model.pt"  --out-csv   "D:\\project\\dino\\cloud\\data\\bkg_mask\\inference_result\\2025.csv"  --batch-size 32 --seed 42
"""
import os
import csv
import json
import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


# ----------------- 复现性 -----------------
def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------- 线性头 -----------------
class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1 if num_classes == 2 else num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


# ----------------- I/O -----------------
def list_images(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    paths = [str(p) for p in Path(folder).glob("*") if p.suffix.lower() in exts]
    paths.sort()
    return paths


def read_existing(csv_path: str) -> set:
    done = set()
    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        return done
    with open(p, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        key = "path" if "path" in reader.fieldnames else "filename"
        for row in reader:
            if key in row:
                done.add(row[key])
    return done


def open_csv_append(csv_path: str, header: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = True
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        write_header = False
    f = open(csv_path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
        f.flush(); os.fsync(f.fileno())
    return f, writer


# ----------------- Checkpoint 兼容加载 -----------------
def load_checkpoint_safe(model_pt: str):
    """
    PyTorch 2.6 起 torch.load 默认 weights_only=True 会限制反序列化，导致包含 numpy 对象的 ckpt 读取失败。
    这里显式设置 weights_only=False；若旧版本不支持该参数，则回退到旧API。
    """
    try:
        blob = torch.load(model_pt, map_location="cpu", weights_only=False)  # 2.6+
    except TypeError:
        # 旧版本 torch 没有 weights_only 参数
        blob = torch.load(model_pt, map_location="cpu")
    return blob


# ----------------- DINOv3 抽特征（生成器：边抽边用） -----------------
@torch.inference_mode()
def extract_features(
    paths: List[str],
    processor: AutoImageProcessor,
    model: AutoModel,
    batch_size: int,
    target_size: int,
    device: str,
    use_cls: bool,
    l2_normalize: bool,
):
    num_register = getattr(model.config, "num_register_tokens", 0)

    def pil_loader(p):
        return Image.open(p).convert("RGB")

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        imgs, ok_idx = [], []
        for j, p in enumerate(batch_paths):
            try:
                imgs.append(pil_loader(p))
                ok_idx.append(i + j)
            except (UnidentifiedImageError, OSError) as e:
                yield "error", (i + j, p, str(e))
        if not imgs:
            continue

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
            feat = hidden[:, 0, :]
        else:
            feat = hidden[:, 1 + num_register:, :].mean(dim=1)

        if l2_normalize:
            feat = F.normalize(feat, dim=1)

        yield "batch", (np.asarray(ok_idx, dtype=np.int64), feat.cpu().numpy())
    yield "done", None


# ----------------- 主流程 -----------------
def main():
    parser = argparse.ArgumentParser(description="DINOv3 + 线性分类器推理（断点续跑版，修复torch.load问题）")
    parser.add_argument("--input-dir", required=True, help="输入图片文件夹")
    parser.add_argument("--model-pt", required=True, help="训练阶段保存的 model.pt")
    parser.add_argument("--out-csv", required=True, help="输出 CSV 路径")
    parser.add_argument("--model-id", default="facebook/dinov3-vitl16-pretrain-lvd1689m",
                        help="DINOv3 模型ID（需与训练时一致）")
    parser.add_argument("--target-size", type=int, default=512)
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls")
    parser.add_argument("--no-l2", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20250101)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-every", type=int, default=1, help="每处理多少个批次强制 flush+fsync")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")

    # 1) 文件列表（排序） & 断点续跑
    all_paths = list_images(args.input_dir)
    if not all_paths:
        raise RuntimeError("输入目录中未找到图片。")
    done = read_existing(args.out_csv)
    paths = [p for p in all_paths if p not in done]
    print(f"Found {len(all_paths)} images; resume: {len(done)} already done, {len(paths)} to infer.")

    # 2) 载入线性模型与标准化参数（安全方式）
    blob = load_checkpoint_safe(args.model_pt)
    # 兼容两种保存方式：训练脚本保存的 dict；或直接保存 state_dict
    if isinstance(blob, dict) and "state_dict" in blob and "in_dim" in blob:
        in_dim = int(blob["in_dim"])
        num_classes = int(blob["num_classes"])
        state_dict = blob["state_dict"]
        norm = blob.get("normalization", None)
        if norm is not None:
            mean = torch.tensor(norm["mean"], dtype=torch.float32).view(1, -1)
            std = torch.tensor(norm["std"], dtype=torch.float32).view(1, -1)
        else:
            mean = torch.zeros(1, in_dim, dtype=torch.float32)
            std = torch.ones(1, in_dim, dtype=torch.float32)
            print("WARN: checkpoint 未包含 normalization，将不做 z-score。")
    else:
        # 只有 state_dict：则无法得知 in_dim/num_classes/mean/std，需要用户保证一致
        raise RuntimeError("模型文件不包含必要的元数据（state_dict/in_dim/num_classes/normalization）。请使用训练脚本生成的 model.pt。")

    lin = LinearClassifier(in_dim, num_classes)
    lin.load_state_dict(state_dict)
    lin.eval().to(device)

    # 3) DINOv3 backbone
    print("Loading DINOv3 backbone...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    backbone = AutoModel.from_pretrained(args.model_id).eval().to(device)

    # 4) CSV & 错误日志
    header = ["index", "path", "pred"] + ([f"prob_{i}" for i in range(num_classes)] if num_classes > 2 else ["prob_1"])
    fcsv, writer = open_csv_append(args.out_csv, header)
    ferr_path = os.path.join(os.path.dirname(args.out_csv) or ".", "errors.txt")
    ferr = open(ferr_path, "a", encoding="utf-8")

    # 5) 推理
    use_cls = (args.pool == "cls")
    l2_norm = not args.no_l2
    pbar = tqdm(total=len(paths), ncols=100, desc="Infer")
    batch_count = 0

    gen = extract_features(
        paths, processor, backbone,
        batch_size=args.batch_size, target_size=args.target_size,
        device=device, use_cls=use_cls, l2_normalize=l2_norm
    )

    for tag, payload in gen:
        if tag == "error":
            _, bad_path, msg = payload
            ferr.write(f"{bad_path}\t{msg}\n"); ferr.flush()
            pbar.update(1)
            continue
        if tag == "done":
            break
        if tag == "batch":
            indices, feat_np = payload
            feat = torch.from_numpy(feat_np).float()
            # 标准化（与训练一致：抽特征 -> L2 -> z-score）
            feat = (feat - mean) / std

            with torch.inference_mode():
                logits = lin(feat.to(device))
                if num_classes == 2:
                    prob = torch.sigmoid(logits.view(-1)).cpu().numpy()
                    pred = (prob >= 0.5).astype(np.int64)
                else:
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    pred = prob.argmax(axis=1)

            for k, idx in enumerate(indices):
                path = paths[int(idx)]
                if num_classes == 2:
                    writer.writerow([int(idx), path, int(pred[k]), float(prob[k])])
                else:
                    writer.writerow([int(idx), path, int(pred[k]), *[float(x) for x in prob[k]]])
                pbar.update(1)

            batch_count += 1
            if batch_count % max(1, args.save_every) == 0:
                fcsv.flush(); os.fsync(fcsv.fileno())

    # 6) 收尾
    fcsv.flush(); os.fsync(fcsv.fileno()); fcsv.close()
    ferr.close(); pbar.close()

    meta = {
        "input_dir": os.path.abspath(args.input_dir),
        "model_pt": os.path.abspath(args.model_pt),
        "out_csv": os.path.abspath(args.out_csv),
        "model_id": args.model_id,
        "target_size": args.target_size,
        "pool": args.pool,
        "l2_normalize": l2_norm,
        "num_classes": num_classes,
        "seed": args.seed
    }
    meta_path = os.path.join(os.path.dirname(args.out_csv) or ".", "infer_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("==> Inference finished.")
    print("CSV:", args.out_csv)
    print("Errors:", ferr_path)
    print("Meta:", meta_path)


if __name__ == "__main__":
    main()