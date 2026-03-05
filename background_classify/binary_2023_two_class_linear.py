# train_linear_pytorch.py
# -*- coding: utf-8 -*-
"""
Reads features extracted by DINOv3, trains and evaluates a PyTorch linear model;

Input:
  X_train.npy, y_train.npy, X_test.npy, y_test.npy
(Optional)paths_train.txt, paths_test.txt, meta.json

Output:
  model.pt                —— 权重+标准化参数+超参
  eval.json               —— 各类指标（accuracy/precision/recall/F1/混淆矩阵等）
  test_predictions.csv    —— 每个样本的路径/真值/概率/预测
  train_log.csv           —— 每个epoch的训练损失


python binary_2023_two_class_linear.py  --feat-dir "D:\\project\\dino\\cloud\\data\\bkg_mask\\features"   --out "D:\\project\\dino\\cloud\\data\\bkg_mask\\lincls"  --epochs 300 --batch-size 256 --lr 0.05 --seed 42
"""

import os
import csv
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# sklearn 可选：若不存在则跳过 AUC/更细指标
try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, roc_auc_score
    )
    SK_OK = True
except Exception:
    SK_OK = False


# ---------- 可复现性 ----------
def set_seed(seed: int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------- 数据 ----------
def load_split(feat_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    feat_dir = Path(feat_dir)
    X_tr = np.load(feat_dir / "X_train.npy")
    y_tr = np.load(feat_dir / "y_train.npy")
    X_te = np.load(feat_dir / "X_test.npy")
    y_te = np.load(feat_dir / "y_test.npy")

    meta = {}
    meta_path = feat_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return X_tr, y_tr, X_te, y_te, meta


def standardize_train_test(X_tr: np.ndarray, X_te: np.ndarray):
    mu = X_tr.mean(axis=0, keepdims=True)
    std = X_tr.std(axis=0, keepdims=True)
    std = np.clip(std, 1e-6, None)
    X_trn = (X_tr - mu) / std
    X_tst = (X_te - mu) / std
    return X_trn, X_tst, mu.squeeze().astype(np.float32), std.squeeze().astype(np.float32)


# ---------- 模型 ----------
class LinearClassifier(nn.Module):
    """二类：输出1个logit；多类：输出C个logit"""
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1 if num_classes == 2 else num_classes, bias=True)

    def forward(self, x):
        return self.linear(x)


# ---------- 评估 ----------
@torch.inference_mode()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    logits_list, y_list = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        logits_list.append(logits.cpu())
        y_list.append(yb)

    logits = torch.cat(logits_list, dim=0)
    y_true = torch.cat(y_list, dim=0).numpy()

    if num_classes == 2:
        # 概率
        prob = torch.sigmoid(logits.view(-1)).numpy()
        y_pred = (prob >= 0.5).astype(np.int64)
    else:
        prob = torch.softmax(logits, dim=1).numpy()
        y_pred = prob.argmax(axis=1)

    # 指标
    acc = float((y_pred == y_true).mean())

    metrics = {"accuracy": acc}
    if SK_OK:
        if num_classes == 2:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred).tolist()
            try:
                auc = roc_auc_score(y_true, prob)
            except Exception:
                auc = None
            metrics.update({
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "auc": float(auc) if auc is not None else None,
                "confusion_matrix": cm,
            })
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred).tolist()
            metrics.update({
                "precision_macro": float(prec),
                "recall_macro": float(rec),
                "f1_macro": float(f1),
                "confusion_matrix": cm,
            })
    return metrics, (y_true, y_pred, prob)


# ---------- 训练 ----------
def train(
    X_tr, y_tr, X_te, y_te, out_dir: str, epochs=100, batch_size=256,
    lr=0.05, weight_decay=0.0, seed=42, device="cuda"
):
    set_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    # 标准化
    X_tr, X_te, mu, std = standardize_train_test(X_tr, X_te)
    in_dim = X_tr.shape[1]

    # 转成 tensor
    X_tr_t = torch.from_numpy(X_tr).float()
    y_tr_t = torch.from_numpy(y_tr.astype(np.int64))
    X_te_t = torch.from_numpy(X_te).float()
    y_te_t = torch.from_numpy(y_te.astype(np.int64))

    num_classes = int(len(np.unique(y_tr)))
    model = LinearClassifier(in_dim, num_classes).to(device)
    model.train()

    # 损失函数
    if num_classes == 2:
        # 处理可能的类不均衡（pos_weight = neg/pos）
        n_pos = float((y_tr == 1).sum())
        n_neg = float((y_tr == 0).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=batch_size, shuffle=False, drop_last=False)

    # 训练循环
    log_path = os.path.join(out_dir, "train_log.csv")
    with open(log_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "train_loss"])

        for epoch in range(1, epochs + 1):
            model.train()
            running = 0.0
            pbar = tqdm(train_loader, ncols=90, desc=f"Epoch {epoch}/{epochs}")
            for xb, yb in pbar:
                xb = xb.to(device)
                if num_classes == 2:
                    yb = yb.float().to(device)
                    logits = model(xb).view(-1)
                    loss = criterion(logits, yb)
                else:
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                running += loss.item() * xb.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running / len(train_loader.dataset)
            writer.writerow([epoch, f"{train_loss:.6f}"])

    # 评估
    metrics, (y_true, y_pred, prob) = evaluate(model, test_loader, device, num_classes)

    # 保存预测
    pred_csv = os.path.join(out_dir, "test_predictions.csv")
    paths_te = None
    # 如果特征目录中有 paths_test.txt，尝试读取以便保存可读路径
    # 这里不访问文件系统之外的路径，由调用方保证路径正确
    # （在 main() 中会设置）
    if hasattr(train, "paths_test"):
        paths_te = train.paths_test

    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if num_classes == 2:
            writer.writerow(["index", "path", "label", "prob_1", "pred"])
            for i in range(len(y_true)):
                pth = "" if paths_te is None else paths_te[i]
                writer.writerow([i, pth, int(y_true[i]), float(prob[i]), int(y_pred[i])])
        else:
            writer.writerow(["index", "path", "label", "pred"])
            for i in range(len(y_true)):
                pth = "" if paths_te is None else paths_te[i]
                writer.writerow([i, pth, int(y_true[i]), int(y_pred[i])])

    # 保存指标
    eval_json = os.path.join(out_dir, "eval.json")
    with open(eval_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 保存权重（包含标准化参数与必要元数据）
    model_blob = {
        "state_dict": model.state_dict(),
        "in_dim": in_dim,
        "num_classes": num_classes,
        "normalization": {"mean": mu, "std": std},
        "args": {
            "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "weight_decay": weight_decay, "seed": seed
        },
    }
    torch.save(model_blob, os.path.join(out_dir, "model.pt"))

    print("==> Done")
    print("Metrics:", metrics)
    print(f"Saved: {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train & evaluate a PyTorch linear classifier on saved features.")
    parser.add_argument("--feat-dir", required=True, help="包含X_train.npy等文件的目录")
    parser.add_argument("--out", required=True, help="输出目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda"
    print("Device:", device)

    X_tr, y_tr, X_te, y_te, meta = load_split(args.feat_dir)

    # 暂存测试路径（若存在）
    feat_dir = Path(args.feat_dir)
    paths_test_file = feat_dir / "paths_test.txt"
    if paths_test_file.exists():
        with open(paths_test_file, "r", encoding="utf-8") as f:
            paths_te = [ln.strip() for ln in f.readlines()]
        # 给 train() 挂一个属性，仅用于写CSV（避免修改函数签名）
        train.paths_test = paths_te  # type: ignore

    train(
        X_tr, y_tr, X_te, y_te,
        out_dir=args.out, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, weight_decay=args.weight_decay, seed=args.seed, device=device
    )


if __name__ == "__main__":
    main()
