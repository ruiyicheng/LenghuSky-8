from __future__ import annotations

# ===========================================
# Combined bootstrap + training (single file)
# ===========================================
# Key changes from your originals:
# - No subprocess calls. Training/eval runs in-process.
# - No copying files into train/val/test folders. We split lists in-memory.
# - Images, masks (downsampled to patch grid), and DINO tokens are all
#   loaded/computed ONCE into RAM, then reused for every seed/config.
# - We still write split manifests (train.txt/val.txt/test.txt) for traceability.
# - We still save per-seed model and test_results JSON like before.
#
# Assumptions (validated at runtime):
# - All base configs share the same {model_id, target_resolution, patch_size},
#   class_map, active_class_names. (global_feature_mode can differ.)
#   If not, the script will assert, because per-model/tokenization would defeat
#   the “load once” goal.
import contextlib
import os
import io
import csv
import json
import math
import time
import base64
import hashlib
import random
import shutil
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import confusion_matrix, classification_report

# ------------------ Paths / run control ------------------

DATA_DIR   = Path(r"D:\project\dino\cloud\data\segmentation\data")          # contains *.json (LabelMe)
SPLIT_ROOT = Path(r"D:\project\dino\cloud\data\segmentation")               # where we write train.txt/val.txt/test.txt manifests
LOG_DIR    = Path(r"D:\project\dino\cloud\log\bootstrap_logs")
METRICS_CSV = Path(r"D:\project\dino\cloud\log\bootstrap_metrics.csv")

# Base configs (differ mainly by global_feature_mode)
CONFIG_DIR    = Path(r"D:\project\dino\cloud\full_mark_with_global_experiment")
CONFIG_NAMES  = ["all_feature", "mean_feature", "cls_feature", "none_feature"]
CONFIG_PATHS  = {name: CONFIG_DIR / f"{name}.json" for name in CONFIG_NAMES}
BOOTSTRAP_CFG_DIR = CONFIG_DIR / "bootstrap_configs"  # saved seeded snapshots (for reference/debug)

# Split / bootstrap
VAL_RATIO  = 0.1   # 8:1:1
TEST_RATIO = 0.1
RECURSIVE  = True
N_BOOTSTRAPS = 20
SEEDS = [10 + i * 17 for i in range(5,N_BOOTSTRAPS)]  # deterministic

# ------------------ Shared config dataclass ------------------

@dataclass
class CONFIG:
    # Input roots (kept for compatibility; we won't traverse them—splits happen in RAM)
    train_labelme_root: str
    test_labelme_root: str
    val_labelme_root: str

    # Output dirs
    cache_dir: str
    coef_out_dir: str
    inference_out_dir: str

    # DINOv3
    model_id: str
    target_resolution: int
    patch_size: int

    # Classes
    class_map: Dict[str, int]        # label -> original id
    active_class_names: List[str]    # subset order defines class index order

    # Global feature concat mode: "all" | "cls" | "mean" | "none"
    global_feature_mode: str

    # PyTorch linear probe
    random_state: int
    torch_batch_size: int
    torch_epochs: int
    torch_optimizer: str         # "adam", "sgd", or "lbfgs"
    torch_lr: float
    torch_weight_decay: float
    torch_momentum: float
    torch_lbfgs_lr: float
    torch_patience: int

    # (unused in merged flow; we split by seed ratios, not json_val_ratio)
    json_val_ratio: float

def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))

def save_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_config(path: Path) -> CONFIG:
    d = load_json(path)
    cfg = CONFIG(**d)
    cfg.global_feature_mode = str(cfg.global_feature_mode).lower()
    assert cfg.global_feature_mode in {"all", "cls", "mean", "none"}, \
        "global_feature_mode must be one of {'all','cls','mean','none'}"
    # We don't use json_val_ratio here, but keep the sanity range:
    assert 0.0 < cfg.json_val_ratio < 0.5, "json_val_ratio should be in (0, 0.5)."
    # Ensure outs exist
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.coef_out_dir, exist_ok=True)
    os.makedirs(cfg.inference_out_dir, exist_ok=True)
    return cfg

# ------------------ LabelMe I/O + rasterization ------------------

@dataclass
class Shape:
    label: str
    shape_type: str
    points: List[Tuple[float, float]]

def _read_image_from_labelme_data(data: dict, json_dir: str) -> Image.Image:
    image_bytes_b64 = data.get("imageData")
    if image_bytes_b64:
        if isinstance(image_bytes_b64, str) and image_bytes_b64.startswith("data:"):
            image_bytes_b64 = image_bytes_b64.split(",", 1)[-1]
        img_bytes = base64.b64decode(image_bytes_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    image_path = data.get("imagePath", "")
    if image_path:
        candidate = image_path if os.path.isabs(image_path) else os.path.join(json_dir, image_path)
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB")
    raise FileNotFoundError("No embedded imageData and no resolvable imagePath in LabelMe JSON.")

def read_labelme_json(json_path: str) -> Tuple[Image.Image, int, int, List[Shape]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    img = _read_image_from_labelme_data(data, os.path.dirname(json_path))
    w = int(data.get("imageWidth", img.width) or img.width)
    h = int(data.get("imageHeight", img.height) or img.height)
    shapes: List[Shape] = []
    for sh in data.get("shapes", []) or []:
        label = (sh.get("label", "") or "").strip()
        shape_type = (sh.get("shape_type", "polygon") or "polygon").lower()
        raw_points = sh.get("points", []) or []
        pts = [(float(x), float(y)) for x, y in raw_points]
        if shape_type == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            pts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            shape_type = "polygon"
        shapes.append(Shape(label=label, shape_type=shape_type, points=pts))
    return img, w, h, shapes

def _circle_bbox_from_two_points(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float, float, float]:
    cx, cy = p0
    px, py = p1
    r = math.hypot(px - cx, py - cy)
    return (cx - r, cy - r, cx + r, cy + r)

def rasterize_mask(
    width: int,
    height: int,
    shapes: List[Shape],
    class_map: Dict[str, int],
    line_width: int = 1,
) -> np.ndarray:
    bg_value = 255
    mask_img = Image.new("L", (width, height), color=bg_value)
    draw = ImageDraw.Draw(mask_img)
    for sh in shapes:
        lbl = (sh.label or "").strip().lower()
        if lbl not in class_map:
            continue
        cls_id = int(class_map[lbl])
        st = sh.shape_type.lower()
        pts = sh.points
        if st in {"polygon"} and len(pts) >= 3:
            draw.polygon(pts, fill=cls_id)
        elif st in {"linestrip", "polyline"} and len(pts) >= 2:
            draw.line(pts, fill=cls_id, width=line_width)
        elif st == "line" and len(pts) == 2:
            draw.line(pts, fill=cls_id, width=line_width)
        elif st == "point" and len(pts) == 1:
            x, y = pts[0]
            draw.point((x, y), fill=cls_id)
        elif st == "circle" and len(pts) == 2:
            bbox = _circle_bbox_from_two_points(pts[0], pts[1])
            draw.ellipse(bbox, fill=cls_id)
        elif st == "rectangle" and len(pts) == 2:
            (x1, y1), (x2, y2) = pts
            x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
            y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
            draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill=cls_id)
        else:
            if len(pts) >= 3:
                draw.polygon(pts, fill=cls_id)
    return np.array(mask_img, dtype=np.int64)

# ------------------ DINO extraction ------------------

@dataclass
class DinoOutputs:
    patch_tokens: np.ndarray  # (N_patches, D)
    global_tokens: np.ndarray # (5, D)
    grid_hw: Tuple[int, int]  # (H_patches, W_patches)

def get_dino_tokens(
    img: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModel,
    target_resolution: int,
    patch_size: int,
    device: torch.device,
) -> DinoOutputs:
    target = (target_resolution // patch_size) * patch_size
    inputs = processor(images=img, size={"height": target, "width": target}, do_center_crop=False, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    hs = out.last_hidden_state  # (1, 1+4+N, D)
    glob = hs[:, :5, :].squeeze(0).detach().cpu().numpy()     # (5, D)
    patch = hs[:, 5:, :].squeeze(0).detach().cpu().numpy()    # (N, D)
    n_tokens, _ = patch.shape
    side = int(round(math.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise RuntimeError(f"Expected square patch grid; got {n_tokens} tokens")
    return DinoOutputs(patch_tokens=patch, global_tokens=glob, grid_hw=(side, side))

def build_concat_features(
    patch_tokens: np.ndarray,
    global_tokens: np.ndarray,
    mode: str = "all",
) -> np.ndarray:
    D = patch_tokens.shape[1]
    mode = (mode or "all").lower()
    if mode == "none":
        feats = patch_tokens
    elif mode == "cls":
        g = global_tokens[0]
        g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)
    elif mode == "mean":
        g = global_tokens.mean(axis=0)
        g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)
    else:  # "all"
        g = global_tokens.reshape(-1)  # (5D,)
        g_rep = np.broadcast_to(g, (patch_tokens.shape[0], 5 * D))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)
    return feats.astype(np.float32)

# ------------------ Dataset discovery ------------------

def gather_files(root: Path, recursive: bool) -> List[Path]:
    if recursive:
        return sorted([p for p in root.rglob("*.json") if p.is_file()])
    else:
        return sorted([p for p in root.glob("*.json") if p.is_file()])

# ------------------ Label mapping helpers ------------------

def _build_label_mapping(cfg: CONFIG) -> Tuple[List[int], Dict[int, int], List[str]]:
    valid_names = list(cfg.active_class_names)
    valid_ids = [cfg.class_map[n] for n in valid_names]
    id_to_new = {old: i for i, old in enumerate(valid_ids)}
    return valid_ids, id_to_new, valid_names

# ------------------ Torch linear probe ------------------

class TorchLinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# def train_linear_probe_torch(
#     X_train: np.ndarray,
#     y_train: np.ndarray,
#     X_val: Optional[np.ndarray],
#     y_val: Optional[np.ndarray],
#     n_classes: int,
#     cfg: CONFIG,
#     device: torch.device,
# ):
#     in_dim = X_train.shape[1]
#     model = TorchLinearProbe(in_dim, n_classes).to(device)

#     # Data
#     xtr = torch.from_numpy(X_train)
#     ytr = torch.from_numpy(y_train.astype(np.int64))
#     train_loader = DataLoader(
#         TensorDataset(xtr, ytr),
#         batch_size=cfg.torch_batch_size,
#         shuffle=False,
#         drop_last=False,
#         num_workers=0  # keep 0 to avoid extra CPU RAM copies when huge
#     )

#     val_loader = None
#     if X_val is not None and y_val is not None and len(X_val) > 0:
#         xv = torch.from_numpy(X_val)
#         yv = torch.from_numpy(y_val.astype(np.int64))
#         val_loader = DataLoader(
#             TensorDataset(xv, yv),
#             batch_size=cfg.torch_batch_size,
#             shuffle=False,
#             num_workers=0
#         )

#     # Optimizer
#     opt_name = cfg.torch_optimizer.lower()
#     if opt_name == "sgd":
#         opt = torch.optim.SGD(model.parameters(), lr=cfg.torch_lr, momentum=cfg.torch_momentum, weight_decay=cfg.torch_weight_decay)
#     elif opt_name == "lbfgs":
#         opt = torch.optim.LBFGS(model.parameters(), lr=cfg.torch_lbfgs_lr, max_iter=100, history_size=50, line_search_fn="strong_wolfe")
#     else:
#         opt = torch.optim.Adam(model.parameters(), lr=cfg.torch_lr, weight_decay=cfg.torch_weight_decay)

#     criterion = nn.CrossEntropyLoss()

#     best_val = float("inf")
#     best_state = None
#     no_improve = 0

#     def run_epoch(loader, train=True):
#         model.train(train)
#         total_loss, total_correct, total_n = 0.0, 0, 0
#         if isinstance(opt, torch.optim.LBFGS) and train:
#             def closure():
#                 opt.zero_grad(set_to_none=True)
#                 loss_accum = 0.0
#                 for xb, yb in loader:
#                     xb = xb.to(device)
#                     yb = yb.to(device)
#                     logits = model(xb)
#                     loss = criterion(logits, yb)
#                     loss.backward()
#                     loss_accum += loss.item() * xb.size(0)
#                 return torch.tensor(loss_accum / len(loader.dataset), device=device, requires_grad=True)
#             _ = opt.step(closure)
#             model.eval()
#             with torch.no_grad():
#                 for xb, yb in loader:
#                     xb = xb.to(device); yb = yb.to(device)
#                     logits = model(xb)
#                     loss = criterion(logits, yb)
#                     total_loss += loss.item() * xb.size(0)
#                     pred = logits.argmax(dim=1)
#                     total_correct += (pred == yb).sum().item()
#                     total_n += xb.size(0)
#             return total_loss / total_n, total_correct / total_n
#         else:
#             for xb, yb in loader:
#                 xb = xb.to(device); yb = yb.to(device)
#                 if train:
#                     opt.zero_grad(set_to_none=True)
#                 logits = model(xb)
#                 loss = criterion(logits, yb)
#                 if train:
#                     loss.backward()
#                     opt.step()
#                 total_loss += loss.item() * xb.size(0)
#                 pred = logits.argmax(dim=1)
#                 total_correct += (pred == yb).sum().item()
#                 total_n += xb.size(0)
#             return total_loss / max(1, total_n), total_correct / max(1, total_n)

#     for epoch in range(1, cfg.torch_epochs + 1):
#         t0 = time.time()
#         tr_loss, tr_acc = run_epoch(train_loader, train=True)
#         if val_loader is not None:
#             model.eval()
#             with torch.no_grad():
#                 val_loss, correct_val, n_val = 0.0, 0, 0
#                 for xb, yb in val_loader:
#                     xb = xb.to(device); yb = yb.to(device)
#                     logits = model(xb)
#                     loss = criterion(logits, yb)
#                     val_loss += loss.item() * xb.size(0)
#                     correct_val += (logits.argmax(1) == yb).sum().item()
#                     n_val += xb.size(0)
#             val_loss /= max(1, n_val)
#             val_acc = correct_val / max(1, n_val)
#         else:
#             val_loss, val_acc = tr_loss, tr_acc

#         dt = time.time() - t0
#         print(f"[torch] epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
#               f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.2f}s")

#         if val_loss + 1e-6 < best_val:
#             best_val = val_loss
#             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
#             no_improve = 0
#         else:
#             no_improve += 1
#             if no_improve >= cfg.torch_patience:
#                 print("Early stopping.")
#                 break

#     if best_state is not None:
#         model.load_state_dict(best_state)

#     return model
def train_linear_probe_torch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val: Optional[np.ndarray],
    n_classes: int,
    cfg: CONFIG,
    device: torch.device,
):
    in_dim = X_train.shape[1]
    model = TorchLinearProbe(in_dim, n_classes).to(device)

    # Try to keep everything on GPU (fast path). Fall back to pinned DataLoader if OOM.
    def _to_gpu(arr, dtype=None):
        t = torch.from_numpy(arr if dtype is None else arr.astype(dtype, copy=False))
        return t.to(device, non_blocking=True)

    use_loader_fallback = False
    try:
        Xtr_dev = _to_gpu(X_train, dtype=np.float32)
        ytr_dev = _to_gpu(y_train.astype(np.int64))
        Xv_dev, yv_dev = None, None
        if X_val is not None and len(X_val) > 0:
            Xv_dev = _to_gpu(X_val, dtype=np.float32)
            yv_dev = _to_gpu(y_val.astype(np.int64))
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            use_loader_fallback = True
            print("Falling back to DataLoader due to OOM:", str(e))
        else:
            raise

    opt_name = cfg.torch_optimizer.lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.torch_lr, momentum=cfg.torch_momentum, weight_decay=cfg.torch_weight_decay)
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=cfg.torch_lbfgs_lr, max_iter=100, history_size=50, line_search_fn="strong_wolfe")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.torch_lr, weight_decay=cfg.torch_weight_decay)

    criterion = nn.CrossEntropyLoss()
    best_val = float("inf"); best_state = None; no_improve = 0

    # Mixed precision for GEMMs
    use_amp = (device.type == "cuda")
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    scaler = torch.cuda.amp.GradScaler(enabled=False)  # not needed for simple linear probe, keep False

    # Helper: evaluate on device tensors
    @torch.no_grad()
    def _eval_dev(Xd, yd):
        if Xd is None or yd is None or Xd.shape[0] == 0:
            return 0.0, 0.0
        bs = max(cfg.torch_batch_size, 8192)
        total_loss, total_correct, total_n = 0.0, 0, 0
        model.eval()
        with amp_ctx:
            for i in range(0, Xd.shape[0], bs):
                xb = Xd[i:i+bs]; yb = yd[i:i+bs]
                logits = model(xb)
                loss = criterion(logits, yb)
                total_loss += loss.item() * xb.size(0)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += xb.size(0)
        return total_loss / max(1, total_n), total_correct / max(1, total_n)

    if not use_loader_fallback:
        # ===== FAST PATH: full GPU tensors with big batches =====
        bs = max(cfg.torch_batch_size, 32768)  # crank this up when fully on device
        for epoch in range(1, cfg.torch_epochs + 1):
            model.train()
            t0 = time.time()
            total_loss, total_correct, total_n = 0.0, 0, 0
            for i in range(0, Xtr_dev.shape[0], bs):
                xb = Xtr_dev[i:i+bs]; yb = ytr_dev[i:i+bs]
                opt.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += xb.size(0)

            tr_loss = total_loss / max(1, total_n)
            tr_acc  = total_correct / max(1, total_n)
            val_loss, val_acc = _eval_dev(Xv_dev, yv_dev) if Xv_dev is not None else (tr_loss, tr_acc)
            dt = time.time() - t0
            print(f"[torch-fast] epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.2f}s")

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.torch_patience:
                    print("Early stopping.")
                    break
        # release GPU memory

    else:
        # ===== FALLBACK: pinned DataLoader (OOM-safe) =====
        xtr = torch.from_numpy(X_train)
        ytr = torch.from_numpy(y_train.astype(np.int64))
        train_loader = DataLoader(
            TensorDataset(xtr, ytr),
            batch_size=cfg.torch_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers= max(4, (os.cpu_count() or 8)//2),
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = None
        if X_val is not None and len(X_val) > 0:
            xv = torch.from_numpy(X_val); yv = torch.from_numpy(y_val.astype(np.int64))
            val_loader = DataLoader(
                TensorDataset(xv, yv),
                batch_size=cfg.torch_batch_size,
                shuffle=False,
                num_workers=max(4, (os.cpu_count() or 8)//2),
                pin_memory=True,
                persistent_workers=True,
            )

        for epoch in range(1, cfg.torch_epochs + 1):
            model.train()
            t0 = time.time()
            total_loss, total_correct, total_n = 0.0, 0, 0
            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with amp_ctx:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item() * xb.size(0)
                total_correct += (logits.argmax(1) == yb).sum().item()
                total_n += xb.size(0)
            tr_loss = total_loss / max(1, total_n)
            tr_acc  = total_correct / max(1, total_n)

            if val_loader is not None:
                model.eval()
                with torch.no_grad(), amp_ctx:
                    val_loss, correct_val, n_val = 0.0, 0, 0
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                        logits = model(xb)
                        loss = criterion(logits, yb)
                        val_loss += loss.item() * xb.size(0)
                        correct_val += (logits.argmax(1) == yb).sum().item()
                        n_val += xb.size(0)
                val_loss /= max(1, n_val)
                val_acc = correct_val / max(1, n_val)
            else:
                val_loss, val_acc = tr_loss, tr_acc

            dt = time.time() - t0
            print(f"[torch-fallback] epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.2f}s")

            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.torch_patience:
                    print("Early stopping.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, target_names: List[str], device: torch.device):
#     model.eval()
#     xb = torch.from_numpy(X).to(device)
#     with torch.no_grad():
#         logits = model(xb)
#         pred = logits.argmax(1).cpu().numpy()
#     acc = float((pred == y).mean())
#     print(f"Accuracy: {acc:.4f}")
#     labels = list(range(len(target_names)))
#     cm = confusion_matrix(y, pred, labels=labels)
#     print("Confusion matrix (rows=true, cols=pred):")
#     print(cm)
#     results = classification_report(
#         y, pred, labels=labels, target_names=target_names, digits=4, output_dict=True
#     )
#     # release GPU memory
#     del xb; torch.cuda.empty_cache()
#     return acc, results
def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   target_names: List[str], device: torch.device):
    model.eval()
    bs = 65536  # big but bounded; tune to taste
    n = X.shape[0]
    preds = np.empty(n, dtype=np.int64)
    total_correct = 0

    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(X[i:i+bs]).to(device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            preds[i:i+bs] = pred
            total_correct += (pred == y[i:i+bs]).sum()
            # free chunk ASAP
            del xb, logits

    acc = float(total_correct) / max(1, n)
    labels = list(range(len(target_names)))
    cm = confusion_matrix(y, preds, labels=labels)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    results = classification_report(
        y, preds, labels=labels, target_names=target_names, digits=4, output_dict=True
    )

    # help allocator return blocks after big eval
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return acc, results

def save_linear_probe_torch(model: nn.Module, class_names: List[str], cfg: CONFIG, fname: str = "linear_probe_dinov3_vitl16_torch.pt") -> str:
    out_path = os.path.join(cfg.coef_out_dir, fname)
    meta = {
        "model_id": cfg.model_id,
        "target_resolution": cfg.target_resolution,
        "patch_size": cfg.patch_size,
        "global_feature_mode": cfg.global_feature_mode,
        "active_class_names": class_names,
        "class_map": cfg.class_map,
    }
    torch.save({"state_dict": model.state_dict(), "meta": meta}, out_path)
    print(f"Saved torch linear probe to: {out_path}")
    return out_path

# ------------------ In-RAM dataset cache ------------------

@dataclass
class CachedItem:
    path: str
    gh: int
    gw: int
    patch_tokens: np.ndarray   # (N, D)
    global_tokens: np.ndarray  # (5, D)
    mask_grid_orig_ids: np.ndarray  # (N,) original dataset ids or 255 for background/ignored

def downsample_mask_to_grid(mask_full: np.ndarray, gw: int, gh: int) -> np.ndarray:
    mask_img = Image.fromarray(mask_full.astype(np.int32), mode="I")
    mask_grid = mask_img.resize((gw, gh), resample=Image.NEAREST)
    return np.array(mask_grid, dtype=np.int64).reshape(-1)  # (N,)

def preload_all_items(
    json_paths: List[str],
    cfg_ref: CONFIG,
    device: torch.device,
) -> Tuple[Dict[str, CachedItem], AutoImageProcessor, AutoModel]:
    """
    Load all images, rasterize masks, extract DINO tokens ONCE into RAM.
    Returns path->CachedItem mapping and the loaded processor+model (to reuse).
    """
    print(f"[preload] Loading DINOv3 {cfg_ref.model_id} ...")
    processor = AutoImageProcessor.from_pretrained(cfg_ref.model_id)
    model = AutoModel.from_pretrained(cfg_ref.model_id).to(device)
    model.eval()

    cache: Dict[str, CachedItem] = {}
    for i, jp in enumerate(json_paths, 1):
        img, w, h, shapes = read_labelme_json(jp)
        # full-res mask in original class ids
        mask_full = rasterize_mask(width=w, height=h, shapes=shapes, class_map=cfg_ref.class_map)
        # tokens
        out = get_dino_tokens(img, processor, model, cfg_ref.target_resolution, cfg_ref.patch_size, device)
        gh, gw = out.grid_hw
        # downsample mask to match patch grid
        y_full = downsample_mask_to_grid(mask_full, gw, gh)  # (N,)
        cache[jp] = CachedItem(
            path=jp,
            gh=gh, gw=gw,
            patch_tokens=out.patch_tokens.astype(np.float32),
            global_tokens=out.global_tokens.astype(np.float32),
            mask_grid_orig_ids=y_full.astype(np.int64),
        )
        if i % 20 == 0 or i == len(json_paths):
            print(f"[preload] processed {i}/{len(json_paths)}")
    return cache, processor, model

def make_split(files: List[str], val_ratio: float, test_ratio: float, seed: int) -> Tuple[List[str], List[str], List[str]]:
    if not (0.0 <= val_ratio < 1.0) or not (0.0 <= test_ratio < 1.0) or (val_ratio + test_ratio >= 1.0):
        raise ValueError("Bad split ratios: ensure 0<=val<1, 0<=test<1, and val+test<1.")
    rng = random.Random(seed)
    shuffled = files.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val  = int(math.floor(n * val_ratio))
    n_test = int(math.floor(n * test_ratio))
    n_train = n - n_val - n_test
    train_files = shuffled[:n_train]
    val_files   = shuffled[n_train:n_train + n_val]
    test_files  = shuffled[n_train + n_val:]
    return train_files, val_files, test_files

def write_manifest(files: List[str], base: Path, out_path: Path) -> None:
    rel_lines = [str(Path(p).resolve().relative_to(base)).replace("\\", "/") for p in files]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rel_lines), encoding="utf-8")

def assemble_Xy_from_cache(
    paths: List[str],
    cfg: CONFIG,
    cache: Dict[str, CachedItem],
) -> Tuple[np.ndarray, np.ndarray]:
    if not paths:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    X_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    valid_ids, id_to_new, _ = _build_label_mapping(cfg)

    for p in paths:
        item = cache[p]
        feats = build_concat_features(item.patch_tokens, item.global_tokens, cfg.global_feature_mode)  # (N, *)
        y_full = item.mask_grid_orig_ids  # (N,)
        keep = np.isin(y_full, valid_ids)
        if not keep.any():
            continue
        X_chunks.append(feats[keep])
        y_chunks.append(np.vectorize(id_to_new.get)(y_full[keep]).astype(np.int64))

    if not X_chunks:
        raise RuntimeError("No usable data in this split after filtering by active_class_names/class_map.")
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    return X, y

def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    BOOTSTRAP_CFG_DIR.mkdir(parents=True, exist_ok=True)

def result_json_path(inference_out_dir: Path, seed: int) -> Path:
    return inference_out_dir / f"test_results_seed_{seed}.json"

def seed_config(base_cfg: CONFIG, seed: int) -> CONFIG:
    # Derive a seed-specific config clone (for metadata/outputs)
    cfg_dict = base_cfg.__dict__.copy()
    cfg_dict["random_state"] = int(seed)
    # inference_out_dir: make seed-specific subfolder
    base_out = base_cfg.inference_out_dir or base_cfg.coef_out_dir or str(CONFIG_DIR)
    base_out_seed = Path(base_out) / f"bootstrap_seed_{seed}"
    cfg_dict["inference_out_dir"] = str(base_out_seed).replace("\\", "/")
    # ensure dirs exist
    os.makedirs(cfg_dict["inference_out_dir"], exist_ok=True)
    os.makedirs(cfg_dict["coef_out_dir"], exist_ok=True)
    # save a JSON snapshot for reproducibility
    dst = BOOTSTRAP_CFG_DIR / f"{Path(base_out).stem}_{Path(base_cfg.model_id).name.replace('/', '_')}_{base_cfg.global_feature_mode}_seed{seed}.json"
    save_json(dst, cfg_dict)
    # return as CONFIG
    return CONFIG(**cfg_dict)

def main():
    ensure_dirs()

    # Load all base configs
    base_cfgs: Dict[str, CONFIG] = {}
    for name, path in CONFIG_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing config JSON: {path}")
        base_cfgs[name] = load_config(path)

    # Validate that model/spec + class taxonomy are identical across configs
    ref = next(iter(base_cfgs.values()))
    for nm, cfg in base_cfgs.items():
        assert (cfg.model_id == ref.model_id
                and cfg.target_resolution == ref.target_resolution
                and cfg.patch_size == ref.patch_size), \
            f"All configs must share model_id/target_resolution/patch_size. Mismatch in {nm}."
        assert cfg.class_map == ref.class_map, f"class_map mismatch in {nm}"
        assert cfg.active_class_names == ref.active_class_names, f"active_class_names mismatch in {nm}"

    # Discover all LabelMe JSONs once
    all_jsons = [str(p) for p in gather_files(DATA_DIR, RECURSIVE)]
    if not all_jsons:
        raise RuntimeError(f"No *.json found under {DATA_DIR} (recursive={RECURSIVE}).")
    print(f"Found {len(all_jsons)} JSONs total.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preload all items (images -> masks -> tokens) ONCE
    cache, processor, model = preload_all_items(all_jsons, ref, device)
    del processor
    if device.type == "cuda":
        model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Prepare CSV header if new
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

        # Iterate seeds
        for seed in SEEDS:
            print(f"\n=== Bootstrap seed={seed} ===")
            train_list, val_list, test_list = make_split(all_jsons, VAL_RATIO, TEST_RATIO, seed)

            # Write split manifests for record (no copying!)
            write_manifest(train_list, DATA_DIR, SPLIT_ROOT / "train.txt")
            write_manifest(val_list,   DATA_DIR, SPLIT_ROOT / "val.txt")
            write_manifest(test_list,  DATA_DIR, SPLIT_ROOT / "test.txt")
            print(f"Split sizes: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

            # For each config variant (mostly global_feature_mode)
            for cfg_name, base_cfg in base_cfgs.items():
                cfg = seed_config(base_cfg, seed)  # adjust random_state + seed-specific out dir

                # Assemble features/labels from in-RAM cache
                print(f"[{cfg_name}] Assembling features (mode={cfg.global_feature_mode}) ...")
                X_tr, y_tr   = assemble_Xy_from_cache(train_list, cfg, cache)
                X_val, y_val = assemble_Xy_from_cache(val_list,   cfg, cache)
                X_te,  y_te  = assemble_Xy_from_cache(test_list,  cfg, cache)
                print(f"[assembled] train_patches={X_tr.shape[0]}, val_patches={X_val.shape[0]}, test_patches={X_te.shape[0]}")

                # Train
                torch.manual_seed(cfg.random_state)
                np.random.seed(cfg.random_state)
                _, _, class_names = _build_label_mapping(cfg)
                n_classes = len(class_names)

                print(f"[{cfg_name}] Training linear probe ...")
                probe = train_linear_probe_torch(X_tr, y_tr, X_val, y_val, n_classes=n_classes, cfg=cfg, device=device)
                save_linear_probe_torch(probe, class_names, cfg)

                # Evaluate on test
                print(f"[{cfg_name}] Evaluating on TEST ...")
                total_accuracy, results = evaluate_torch(probe, X_te, y_te, target_names=class_names, device=device)
                out_json = {"overall_accuracy": total_accuracy, "per_class_results": results}

                # Save per-seed result JSON
                out_path = result_json_path(Path(cfg.inference_out_dir), seed)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(out_json, f, ensure_ascii=False, indent=2)
                print(f"Wrote results: {out_path}")

                # Extract flat metrics for CSV (tolerant to missing keys)
                def pick(cls_name: str, key: str) -> Optional[float]:
                    d = results.get(cls_name, {})
                    v = d.get(key, None)
                    return float(v) if isinstance(v, (int, float, np.floating)) else None

                row = [
                    seed, cfg_name,
                    total_accuracy if isinstance(total_accuracy, (int, float, np.floating)) else "",
                    pick("sky", "precision"), pick("sky", "recall"), pick("sky", "f1-score"),
                    pick("cloud", "precision"), pick("cloud", "recall"), pick("cloud", "f1-score"),
                    pick("contamination", "precision"), pick("contamination", "recall"), pick("contamination", "f1-score"),
                ]
                writer.writerow(row)
                csvfile.flush()

if __name__ == "__main__":

    main()
