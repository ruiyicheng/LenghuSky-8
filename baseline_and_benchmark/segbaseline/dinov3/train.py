# # # """
# # # DINOv3 linear probe on LabelMe annotations (sky/cloud/contamination).

# # # This version:
# # # - **Torch-only** linear probe (no sklearn classifier).
# # # - Uses **global feature concat**: concatenate DINOv3 global tokens with each local patch token.
# # #   Configurable via `CONFIG.global_feature_mode`:
# # #     - "cls":       concat [CLS] token to each patch token  → feature dim = 2*D
# # #     - "mean":      concat mean([CLS] + 4 registers)        → feature dim = 2*D
# # #     - "all":       concat all 5 global tokens (flattened)  → feature dim = 6*D (DEFAULT)
# # # - Supports non-rectangular LabelMe shapes (polygon, line/linestrip, point, **circle**).
# # # - Caches DINOv3 tokens to speed up re-runs.
# # # - **No background class** in training/eval/inference — only {sky, cloud, contamination}.

# # # Usage (edit the CONFIG block), then run:
# # #     python dinov3_linear_probe_labelme_global_concat.py
# # # """
# # # from __future__ import annotations

# # # import os
# # # import csv
# # # import json
# # # import math
# # # import time
# # # import hashlib
# # # from dataclasses import dataclass
# # # from typing import Dict, List, Tuple, Optional

# # # import numpy as np
# # # from PIL import Image, ImageDraw

# # # import torch
# # # import torch.nn as nn
# # # from torch.utils.data import TensorDataset, DataLoader
# # # from transformers import AutoImageProcessor, AutoModel
# # # from sklearn.metrics import confusion_matrix, classification_report
# # # import matplotlib.pyplot as plt

# # # # =============================
# # # # CONFIG — EDIT THESE PATHS
# # # # =============================
# # # class CONFIG:
# # #     # Root dirs
# # #     image_root = r"D:\project\dino\cloud\data\2020"           # directory containing original RGB images
# # #     labelme_root = r"D:\project\dino\cloud\data\2020_label"   # directory containing *.json annotations
# # #     csv_path = r"D:\project\dino\cloud\log\experiment1\train_test_labels.csv"  # CSV with columns: filename,split

# # #     # Output dirs
# # #     cache_dir = r"D:\project\dino\cloud\log\experiment1"              # cached DINO tokens
# # #     coef_out_dir = r"D:\project\dino\cloud\log\experiment1"           # where to save linear probe weights
# # #     inference_out_dir = r"D:\project\dino\cloud\log\experiment1\inference"  # predicted masks/overlays

# # #     # DINOv3
# # #     model_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"  # ViT-L/16
# # #     target_resolution = 512 * 2  # resized (kept multiple of patch size)
# # #     patch_size = 16

# # #     # Classes — map LabelMe labels → integer ids (full set in data)
# # #     class_map = {
# # #         # "background": 0,
# # #         "sky": 0,
# # #         "cloud": 1,
# # #         "contamination": 2,
# # #     }

# # #     # === Only consider these classes for training/eval/inference labels ===
# # #     # Background is excluded on purpose.
# # #     active_class_names = ["sky", "cloud", "contamination"]

# # #     # Global feature concat mode: "cls" | "mean" | "all"
# # #     global_feature_mode = "all"

# # #     # PyTorch linear probe
# # #     random_state = 42
# # #     torch_batch_size = 8192
# # #     torch_epochs = 500
# # #     torch_optimizer = "adam"      # "adam", "sgd", or "lbfgs"
# # #     torch_lr = 1e-3               # for adam/sgd
# # #     torch_weight_decay = 0.0
# # #     torch_momentum = 0.9          # for sgd
# # #     torch_lbfgs_lr = 1.0          # for lbfgs
# # #     torch_patience = 5            # early stopping patience (epochs w/o val improv)

# # # # ensure output dirs exist
# # # os.makedirs(CONFIG.cache_dir, exist_ok=True)
# # # os.makedirs(CONFIG.coef_out_dir, exist_ok=True)
# # # os.makedirs(CONFIG.inference_out_dir, exist_ok=True)

# # # # =============================
# # # # LabelMe reading helpers
# # # # =============================
# # # @dataclass
# # # class Shape:
# # #     label: str
# # #     shape_type: str
# # #     points: List[Tuple[float, float]]


# # # def read_labelme_json(path: str) -> Tuple[str, int, int, List[Shape]]:
# # #     """Return (image_path, image_width, image_height, shapes)."""
# # #     with open(path, "r", encoding="utf-8") as f:
# # #         data = json.load(f)
# # #     image_path = data.get("imagePath", "")
# # #     w = int(data.get("imageWidth", 0) or 0)
# # #     h = int(data.get("imageHeight", 0) or 0)
# # #     shapes: List[Shape] = []
# # #     for sh in data.get("shapes", []) or []:
# # #         label = (sh.get("label", "") or "").strip()
# # #         shape_type = (sh.get("shape_type", "polygon") or "polygon").lower()
# # #         raw_points = sh.get("points", []) or []
# # #         pts = [(float(x), float(y)) for x, y in raw_points]
# # #         if shape_type == "rectangle" and len(pts) == 2:
# # #             (x1, y1), (x2, y2) = pts
# # #             x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
# # #             y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
# # #             pts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
# # #             shape_type = "polygon"
# # #         shapes.append(Shape(label=label, shape_type=shape_type, points=pts))
# # #     return image_path, w, h, shapes


# # # def resolve_image_path(image_root: str, ann_image_path: str) -> str:
# # #     """Try to resolve the absolute image path from LabelMe's imagePath field."""
# # #     if ann_image_path and os.path.isabs(ann_image_path) and os.path.exists(ann_image_path):
# # #         return ann_image_path
# # #     base = os.path.basename(ann_image_path) if ann_image_path else None
# # #     if base:
# # #         candidate = os.path.join(image_root, base)
# # #         if os.path.exists(candidate):
# # #             return candidate
# # #     for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
# # #         candidate = os.path.join(image_root, os.path.splitext(base or "")[0] + ext)
# # #         if os.path.exists(candidate):
# # #             return candidate
# # #     return os.path.join(image_root, base or ann_image_path)

# # # # =============================
# # # # Rasterization: shapes → pixel mask
# # # # =============================

# # # def _circle_bbox_from_two_points(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float, float, float]:
# # #     cx, cy = p0
# # #     px, py = p1
# # #     r = math.hypot(px - cx, py - cy)
# # #     return (cx - r, cy - r, cx + r, cy + r)


# # # def rasterize_mask(
# # #     width: int,
# # #     height: int,
# # #     shapes: List[Shape],
# # #     class_map: Dict[str, int],
# # #     line_width: int = 1,
# # # ) -> np.ndarray:
# # #     """Rasterize LabelMe shapes into an integer mask (H, W) with class ids.

# # #     Later shapes overwrite earlier ones if overlapping. Unknown labels are ignored.
# # #     Supports: polygon/rectangle, line/linestrip, point, **circle** (center + point).
# # #     """
# # #     bg_id = class_map.get("background", 0)
# # #     mask_img = Image.new("L", (width, height), color=bg_id)
# # #     draw = ImageDraw.Draw(mask_img)

# # #     for sh in shapes:
# # #         lbl = (sh.label or "").strip().lower()
# # #         if lbl not in class_map:
# # #             continue
# # #         cls_id = int(class_map[lbl])
# # #         st = sh.shape_type.lower()
# # #         pts = sh.points

# # #         if st in {"polygon"} and len(pts) >= 3:
# # #             draw.polygon(pts, fill=cls_id)
# # #         elif st in {"linestrip", "polyline"} and len(pts) >= 2:
# # #             draw.line(pts, fill=cls_id, width=line_width)
# # #         elif st == "line" and len(pts) == 2:
# # #             draw.line(pts, fill=cls_id, width=line_width)
# # #         elif st == "point" and len(pts) == 1:
# # #             x, y = pts[0]
# # #             draw.point((x, y), fill=cls_id)
# # #         elif st == "circle" and len(pts) == 2:
# # #             bbox = _circle_bbox_from_two_points(pts[0], pts[1])
# # #             draw.ellipse(bbox, fill=cls_id)
# # #         elif st == "rectangle" and len(pts) == 2:
# # #             (x1, y1), (x2, y2) = pts
# # #             x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
# # #             y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
# # #             draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill=cls_id)
# # #         else:
# # #             if len(pts) >= 3:
# # #                 draw.polygon(pts, fill=cls_id)
# # #     return np.array(mask_img, dtype=np.int64)

# # # # =============================
# # # # DINOv3 feature extraction
# # # # =============================

# # # @dataclass
# # # class DinoOutputs:
# # #     patch_tokens: np.ndarray  # (N_patches, D)
# # #     global_tokens: np.ndarray # (5, D) — [CLS] + 4 registers
# # #     grid_hw: Tuple[int, int]  # (H_patches, W_patches)


# # # def get_dino_tokens(
# # #     img: Image.Image,
# # #     processor: AutoImageProcessor,
# # #     model: AutoModel,
# # #     target_resolution: int,
# # #     patch_size: int,
# # #     device: torch.device,
# # # ) -> DinoOutputs:
# # #     """Extract DINOv3 tokens.

# # #     Keeps [CLS] + 4 register tokens **and** patch tokens.
# # #     """
# # #     target = (target_resolution // patch_size) * patch_size
# # #     inputs = processor(images=img, size={"height": target, "width": target}, do_center_crop=False, return_tensors="pt")
# # #     inputs = {k: v.to(device) for k, v in inputs.items()}
# # #     with torch.inference_mode():
# # #         out = model(**inputs)
# # #     hs = out.last_hidden_state  # (1, 1+4+N_patches, D)
# # #     glob = hs[:, :5, :].squeeze(0).cpu().numpy()     # (5, D)
# # #     patch = hs[:, 5:, :].squeeze(0).cpu().numpy()    # (N_patches, D)
# # #     n_tokens, _ = patch.shape
# # #     side = int(math.sqrt(n_tokens))
# # #     assert side * side == n_tokens, f"Expected square grid, got {n_tokens} tokens"
# # #     return DinoOutputs(patch_tokens=patch, global_tokens=glob, grid_hw=(side, side))


# # # def build_concat_features(patch_tokens: np.ndarray, global_tokens: np.ndarray, mode: str = "all") -> np.ndarray:
# # #     """Concatenate global features with local (per-patch) tokens.

# # #     mode:
# # #       - "cls":  concat CLS only → [patch, cls]
# # #       - "mean": concat mean(global 5) → [patch, mean]
# # #       - "all":  concat all 5 global tokens flattened → [patch, glob_flat]
# # #     """
# # #     D = patch_tokens.shape[1]
# # #     if mode == "cls":
# # #         g = global_tokens[0]
# # #         g_rep = np.tile(g, (patch_tokens.shape[0], 1))
# # #         feats = np.concatenate([patch_tokens, g_rep], axis=1)  # (N, 2D)
# # #     elif mode == "mean":
# # #         g = global_tokens.mean(axis=0)
# # #         g_rep = np.tile(g, (patch_tokens.shape[0], 1))
# # #         feats = np.concatenate([patch_tokens, g_rep], axis=1)  # (N, 2D)
# # #     else:  # "all"
# # #         g = global_tokens.reshape(-1)  # (5D,)
# # #         g_rep = np.tile(g, (patch_tokens.shape[0], 1))
# # #         feats = np.concatenate([patch_tokens, g_rep], axis=1)  # (N, 6D)
# # #     return feats.astype(np.float32)

# # # # =============================
# # # # Dataset assembly
# # # # =============================

# # # def load_split(csv_path: str) -> Dict[str, List[str]]:
# # #     splits: Dict[str, List[str]] = {"train": [], "test": []}
# # #     with open(csv_path, "r", newline="", encoding="utf-8") as f:
# # #         reader = csv.DictReader(f)
# # #         for row in reader:
# # #             fn = row["filename"].strip()
# # #             sp = row["split"].strip().lower()
# # #             assert sp in ("train", "test"), f"Unknown split {sp}"
# # #             splits[sp].append(fn)
# # #     return splits


# # # def cache_key(path: str, cfg: CONFIG) -> str:
# # #     h = hashlib.sha256()
# # #     h.update(path.encode("utf-8"))
# # #     h.update(cfg.model_id.encode("utf-8"))
# # #     h.update(str(cfg.target_resolution).encode("utf-8"))
# # #     h.update(cfg.global_feature_mode.encode("utf-8"))
# # #     return h.hexdigest()[:16]


# # # def extract_features_and_labels(
# # #     json_list: List[str],
# # #     cfg: CONFIG,
# # #     processor: AutoImageProcessor,
# # #     model: AutoModel,
# # #     device: torch.device,
# # # ) -> Tuple[np.ndarray, np.ndarray]:
# # #     X_list: List[np.ndarray] = []
# # #     y_list: List[np.ndarray] = []

# # #     for i, json_name in enumerate(json_list, 1):
# # #         ann_path = os.path.join(cfg.labelme_root, json_name)
# # #         if not os.path.exists(ann_path):
# # #             print(f"[WARN] Missing annotation: {ann_path}")
# # #             continue

# # #         ck = cache_key(ann_path, cfg)
# # #         feat_cache = os.path.join(cfg.cache_dir, f"{ck}.npz")

# # #         img_rel, w, h, shapes = read_labelme_json(ann_path)
# # #         img_path = resolve_image_path(cfg.image_root, img_rel)
# # #         if not os.path.exists(img_path):
# # #             print(f"[WARN] Missing image: {img_path}")
# # #             continue

# # #         mask = rasterize_mask(width=w, height=h, shapes=shapes, class_map=cfg.class_map)
# # #         img = Image.open(img_path).convert("RGB")

# # #         if os.path.exists(feat_cache):
# # #             data = np.load(feat_cache)
# # #             patch_tokens = data["patch_tokens"]
# # #             global_tokens = data["global_tokens"]
# # #             gh = int(data["gh"]) if "gh" in data else int(math.sqrt(patch_tokens.shape[0]))
# # #             gw = int(data["gw"]) if "gw" in data else int(math.sqrt(patch_tokens.shape[0]))
# # #         else:
# # #             out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
# # #             patch_tokens = out.patch_tokens
# # #             global_tokens = out.global_tokens
# # #             gh, gw = out.grid_hw
# # #             np.savez_compressed(feat_cache, patch_tokens=patch_tokens, global_tokens=global_tokens, gh=gh, gw=gw)

# # #         # Downsample mask to patch grid using nearest neighbor
# # #         mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
# # #         mask_grid = mask_img.resize((gw, gh), resample=Image.NEAREST)
# # #         y = np.array(mask_grid, dtype=np.int64).reshape(-1)

# # #         X = build_concat_features(patch_tokens, global_tokens, cfg.global_feature_mode)

# # #         X_list.append(X)
# # #         y_list.append(y)

# # #         if i % 20 == 0:
# # #             print(f"Processed {i}/{len(json_list)}: {os.path.basename(img_path)}")

# # #     if not X_list:
# # #         raise RuntimeError("No data loaded — check paths and CSV")

# # #     X = np.concatenate(X_list, axis=0)
# # #     y = np.concatenate(y_list, axis=0)
# # #     return X, y

# # # # =============================
# # # # PyTorch Training / Evaluation
# # # # =============================

# # # class TorchLinearProbe(nn.Module):
# # #     def __init__(self, in_dim: int, n_classes: int):
# # #         super().__init__()
# # #         self.fc = nn.Linear(in_dim, n_classes)

# # #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# # #         return self.fc(x)


# # # def train_linear_probe_torch(
# # #     X_train: np.ndarray,
# # #     y_train: np.ndarray,
# # #     X_val: Optional[np.ndarray],
# # #     y_val: Optional[np.ndarray],
# # #     n_classes: int,
# # #     cfg: CONFIG,
# # #     device: torch.device,
# # # ):
# # #     in_dim = X_train.shape[1]
# # #     model = TorchLinearProbe(in_dim, n_classes).to(device)

# # #     # Data
# # #     xtr = torch.from_numpy(X_train)
# # #     ytr = torch.from_numpy(y_train.astype(np.int64))
# # #     train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.torch_batch_size, shuffle=True, drop_last=False)

# # #     if X_val is not None and y_val is not None:
# # #         xv = torch.from_numpy(X_val)
# # #         yv = torch.from_numpy(y_val.astype(np.int64))
# # #         val_loader = DataLoader(TensorDataset(xv, yv), batch_size=cfg.torch_batch_size, shuffle=False)
# # #     else:
# # #         val_loader = None

# # #     # Optimizer
# # #     if cfg.torch_optimizer.lower() == "sgd":
# # #         opt = torch.optim.SGD(model.parameters(), lr=cfg.torch_lr, momentum=cfg.torch_momentum, weight_decay=cfg.torch_weight_decay)
# # #     elif cfg.torch_optimizer.lower() == "lbfgs":
# # #         opt = torch.optim.LBFGS(model.parameters(), lr=cfg.torch_lbfgs_lr, max_iter=100, history_size=50, line_search_fn="strong_wolfe")
# # #     else:
# # #         opt = torch.optim.Adam(model.parameters(), lr=cfg.torch_lr, weight_decay=cfg.torch_weight_decay)

# # #     criterion = nn.CrossEntropyLoss()

# # #     best_val = float("inf")
# # #     best_state = None
# # #     no_improve = 0

# # #     def run_epoch(loader, train=True):
# # #         model.train(train)
# # #         total_loss, total_correct, total_n = 0.0, 0, 0
# # #         if isinstance(opt, torch.optim.LBFGS) and train:
# # #             def closure():
# # #                 opt.zero_grad(set_to_none=True)
# # #                 loss_accum = 0.0
# # #                 for xb, yb in loader:
# # #                     xb = xb.to(device)
# # #                     yb = yb.to(device)
# # #                     logits = model(xb)
# # #                     loss = criterion(logits, yb)
# # #                     loss.backward()
# # #                     loss_accum += loss.item() * xb.size(0)
# # #                 return torch.tensor(loss_accum / len(loader.dataset), device=device, requires_grad=True)
# # #             _ = opt.step(closure)
# # #             model.eval()
# # #             with torch.no_grad():
# # #                 for xb, yb in loader:
# # #                     xb = xb.to(device)
# # #                     yb = yb.to(device)
# # #                     logits = model(xb)
# # #                     loss = criterion(logits, yb)
# # #                     total_loss += loss.item() * xb.size(0)
# # #                     pred = logits.argmax(dim=1)
# # #                     total_correct += (pred == yb).sum().item()
# # #                     total_n += xb.size(0)
# # #             return total_loss / total_n, total_correct / total_n
# # #         else:
# # #             for xb, yb in loader:
# # #                 xb = xb.to(device)
# # #                 yb = yb.to(device)
# # #                 if train:
# # #                     opt.zero_grad(set_to_none=True)
# # #                 logits = model(xb)
# # #                 loss = criterion(logits, yb)
# # #                 if train:
# # #                     loss.backward()
# # #                     opt.step()
# # #                 total_loss += loss.item() * xb.size(0)
# # #                 pred = logits.argmax(dim=1)
# # #                 total_correct += (pred == yb).sum().item()
# # #                 total_n += xb.size(0)
# # #             return total_loss / total_n, total_correct / total_n

# # #     for epoch in range(1, cfg.torch_epochs + 1):
# # #         t0 = time.time()
# # #         tr_loss, tr_acc = run_epoch(train_loader, train=True)
# # #         if val_loader is not None:
# # #             model.eval()
# # #             with torch.no_grad():
# # #                 val_loss, correct_val, n_val = 0.0, 0, 0
# # #                 for xb, yb in val_loader:
# # #                     xb = xb.to(device)
# # #                     yb = yb.to(device)
# # #                     logits = model(xb)
# # #                     loss = criterion(logits, yb)
# # #                     val_loss += loss.item() * xb.size(0)
# # #                     correct_val += (logits.argmax(1) == yb).sum().item()
# # #                     n_val += xb.size(0)
# # #             val_loss /= max(1, n_val)
# # #             val_acc = correct_val / max(1, n_val)
# # #         else:
# # #             val_loss, val_acc = tr_loss, tr_acc

# # #         dt = time.time() - t0
# # #         print(f"[torch] epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.2f}s")

# # #         if val_loss + 1e-6 < best_val:
# # #             best_val = val_loss
# # #             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
# # #             no_improve = 0
# # #         else:
# # #             no_improve += 1
# # #             if no_improve >= cfg.torch_patience:
# # #                 print("Early stopping.")
# # #                 break

# # #     if best_state is not None:
# # #         model.load_state_dict(best_state)

# # #     return model


# # # def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, target_names: List[str], device: torch.device) -> None:
# # #     model.eval()
# # #     xb = torch.from_numpy(X).to(device)
# # #     with torch.no_grad():
# # #         logits = model(xb)
# # #         pred = logits.argmax(1).cpu().numpy()
# # #     acc = float((pred == y).mean())
# # #     print(f"Accuracy: {acc:.4f}")
# # #     labels = list(range(len(target_names)))
# # #     cm = confusion_matrix(y, pred, labels=labels)
# # #     print("Confusion matrix (rows=true, cols=pred):")
# # #     print(cm)
# # #     print(classification_report(y, pred, labels=labels, target_names=target_names, digits=4))


# # # def save_linear_probe_torch(model: nn.Module, class_names: List[str], cfg: CONFIG, fname: str = "linear_probe_dinov3_vitl16_torch.pt") -> str:
# # #     out_path = os.path.join(cfg.coef_out_dir, fname)
# # #     meta = {
# # #         "model_id": cfg.model_id,
# # #         "target_resolution": cfg.target_resolution,
# # #         "patch_size": cfg.patch_size,
# # #         "global_feature_mode": cfg.global_feature_mode,
# # #         "active_class_names": class_names,
# # #     }
# # #     torch.save({"state_dict": model.state_dict(), "meta": meta}, out_path)
# # #     print(f"Saved torch linear probe to: {out_path}")
# # #     return out_path

# # # # =============================
# # # # Inference & Visualization
# # # # =============================

# # # def predict_patch_grid_torch(
# # #     img: Image.Image,
# # #     cfg: CONFIG,
# # #     processor: AutoImageProcessor,
# # #     model: AutoModel,
# # #     device: torch.device,
# # #     probe: nn.Module,
# # # ) -> Tuple[np.ndarray, Tuple[int, int]]:
# # #     out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
# # #     X = build_concat_features(out.patch_tokens, out.global_tokens, cfg.global_feature_mode)
# # #     X = torch.from_numpy(X).to(device)
# # #     with torch.no_grad():
# # #         logits = probe(X)
# # #         y_pred = logits.argmax(1).cpu().numpy()  # values in [0, n_active_classes-1]
# # #     H, W = out.grid_hw
# # #     return y_pred.reshape(H, W), out.grid_hw


# # # def upsample_to_image(mask_grid: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
# # #     H, W = target_hw
# # #     grid_img = Image.fromarray(mask_grid.astype(np.int32), mode="I")
# # #     up = grid_img.resize((W, H), resample=Image.NEAREST)
# # #     return np.array(up, dtype=np.int64)


# # # def overlay_mask_on_image(
# # #     img: Image.Image,
# # #     mask: np.ndarray,
# # #     alpha: float = 0.5,
# # #     class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
# # #     show: bool = True,
# # #     save_path: Optional[str] = None,
# # # ):
# # #     base = img.convert("RGBA")
# # #     H, W = base.height, base.width

# # #     if class_colors is None:
# # #         # Default palette for *active* classes indexed 0..K-1
# # #         class_colors = {
# # #             0: (0, 114, 178),      # sky (blue-ish)
# # #             1: (230, 159, 0),      # cloud (orange)
# # #             2: (204, 121, 167),    # contamination (pink)
# # #         }

# # #     color_mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))
# # #     for cls_id, rgb in class_colors.items():
# # #         ys, xs = np.where(mask == cls_id)
# # #         if ys.size == 0:
# # #             continue
# # #         layer = Image.new("RGBA", (W, H), (*rgb, int(255 * alpha)))
# # #         m = Image.fromarray((mask == cls_id).astype(np.uint8) * 255, mode="L")
# # #         color_mask = Image.composite(layer, color_mask, m)

# # #     blended = Image.alpha_composite(base, color_mask)

# # #     if show:
# # #         plt.figure(figsize=(8, 8))
# # #         plt.imshow(blended)
# # #         plt.axis('off')
# # #         plt.tight_layout()
# # #         plt.show()

# # #     if save_path:
# # #         blended.convert("RGB").save(save_path)
# # #         print(f"Saved overlay to: {save_path}")

# # #     return blended


# # # def run_inference_on_images(
# # #     image_paths: List[str],
# # #     cfg: CONFIG,
# # #     processor: AutoImageProcessor,
# # #     model: AutoModel,
# # #     device: torch.device,
# # #     probe: nn.Module,
# # #     save_results: bool = True,
# # #     show_plots: bool = False,
# # #     save_overlays: bool = True,
# # # ) -> List[Tuple[str, np.ndarray]]:
# # #     results: List[Tuple[str, np.ndarray]] = []
# # #     for p in image_paths:
# # #         img = Image.open(p).convert("RGB")
# # #         grid_mask, _ = predict_patch_grid_torch(img, cfg, processor, model, device, probe)
# # #         up_mask = upsample_to_image(grid_mask, (img.height, img.width))
# # #         results.append((p, up_mask))

# # #         stem = os.path.splitext(os.path.basename(p))[0]

# # #         if save_results:
# # #             out_mask_path = os.path.join(cfg.inference_out_dir, f"{stem}_mask.png")
# # #             Image.fromarray(up_mask.astype(np.uint8)).save(out_mask_path)
# # #             print(f"Saved mask: {out_mask_path}")

# # #         if save_overlays:
# # #             out_overlay_path = os.path.join(cfg.inference_out_dir, f"{stem}_overlay.jpg")
# # #             overlay_mask_on_image(img, up_mask, alpha=0.5, show=show_plots, save_path=out_overlay_path)

# # #         if show_plots and not save_overlays:
# # #             plt.figure(figsize=(8, 8))
# # #             plt.imshow(img)
# # #             plt.imshow(up_mask, alpha=0.4)
# # #             plt.axis('off')
# # #             plt.tight_layout()
# # #             plt.show()
# # #     return results

# # # # =============================
# # # # Main
# # # # =============================

# # # def _build_label_mapping(cfg: CONFIG) -> Tuple[List[int], Dict[int, int], List[str]]:
# # #     """Return (valid_ids, oldid_to_new, class_names_in_new_index_order)."""
# # #     valid_names = list(cfg.active_class_names)
# # #     valid_ids = [cfg.class_map[n] for n in valid_names]
# # #     id_to_new = {old: i for i, old in enumerate(valid_ids)}
# # #     return valid_ids, id_to_new, valid_names


# # # def _filter_and_remap_labels(y: np.ndarray, valid_ids: List[int], id_to_new: Dict[int, int]):
# # #     keep = np.isin(y, valid_ids)
# # #     y_kept = y[keep]
# # #     y_new = np.vectorize(id_to_new.get)(y_kept)
# # #     return keep, y_new.astype(np.int64)


# # # def main():
# # #     cfg = CONFIG()

# # #     torch.manual_seed(cfg.random_state)
# # #     np.random.seed(cfg.random_state)

# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # #     print(f"Using device: {device}")

# # #     print("Loading DINOv3 model...")
# # #     processor = AutoImageProcessor.from_pretrained(cfg.model_id)
# # #     model = AutoModel.from_pretrained(cfg.model_id).to(device)
# # #     model.eval()

# # #     splits = load_split(cfg.csv_path)
# # #     print({k: len(v) for k, v in splits.items()})

# # #     valid_ids, id_to_new, class_names = _build_label_mapping(cfg)

# # #     # ------- Train data
# # #     print("Extracting train features...")
# # #     X_train, y_train_raw = extract_features_and_labels(splits["train"], cfg, processor, model, device)
# # #     keep_tr, y_train = _filter_and_remap_labels(y_train_raw, valid_ids, id_to_new)
# # #     X_train = X_train[keep_tr]

# # #     # 90/10 train/val split
# # #     n = X_train.shape[0]
# # #     idx = np.arange(n)
# # #     np.random.shuffle(idx)
# # #     n_val = max(1, int(0.1 * n))
# # #     val_idx = idx[:n_val]
# # #     tr_idx = idx[n_val:]
# # #     X_val, y_val = X_train[val_idx], y_train[val_idx]
# # #     X_tr,  y_tr  = X_train[tr_idx],  y_train[tr_idx]

# # #     n_classes = len(class_names)
# # #     probe = train_linear_probe_torch(X_tr, y_tr, X_val, y_val, n_classes=n_classes, cfg=cfg, device=device)
# # #     save_linear_probe_torch(probe, class_names, cfg)

# # #     # ------- Test data
# # #     print("Extracting test features...")
# # #     X_test, y_test_raw = extract_features_and_labels(splits["test"], cfg, processor, model, device)
# # #     keep_te, y_test = _filter_and_remap_labels(y_test_raw, valid_ids, id_to_new)
# # #     X_test = X_test[keep_te]

# # #     # ------- Eval
# # #     evaluate_torch(probe, X_test, y_test, target_names=class_names, device=device)


# # # if __name__ == "__main__":
# # #     main()
# # from __future__ import annotations

# # import os
# # import io
# # import re
# # import csv
# # import json
# # import math
# # import time
# # import base64
# # import hashlib
# # import argparse
# # from dataclasses import dataclass, asdict
# # from typing import Dict, List, Tuple, Optional

# # import numpy as np
# # from PIL import Image, ImageDraw

# # import torch
# # import torch.nn as nn
# # from torch.utils.data import TensorDataset, DataLoader
# # from transformers import AutoImageProcessor, AutoModel
# # from sklearn.metrics import confusion_matrix, classification_report
# # import matplotlib.pyplot as plt

# # # =============================
# # # Config loading
# # # =============================

# # @dataclass
# # class CONFIG:
# #     # Input folders (each contains *.json produced by LabelMe)
# #     train_labelme_root: str
# #     test_labelme_root: str

# #     # Output dirs
# #     cache_dir: str
# #     coef_out_dir: str
# #     inference_out_dir: str

# #     # DINOv3
# #     model_id: str  # e.g., "facebook/dinov3-vitl16-pretrain-lvd1689m"
# #     target_resolution: int  # resized (kept multiple of patch size)
# #     patch_size: int         # 16 for ViT-L/16 etc.

# #     # Classes — map LabelMe labels → integer ids (original ids in data)
# #     class_map: Dict[str, int]

# #     # Only consider these classes for training/eval/inference labels
# #     active_class_names: List[str]

# #     # Global feature concat mode: "all" | "cls" | "mean" | "none"
# #     global_feature_mode: str

# #     # PyTorch linear probe
# #     random_state: int
# #     torch_batch_size: int
# #     torch_epochs: int
# #     torch_optimizer: str         # "adam", "sgd", or "lbfgs"
# #     torch_lr: float              # for adam/sgd
# #     torch_weight_decay: float
# #     torch_momentum: float        # for sgd
# #     torch_lbfgs_lr: float        # for lbfgs
# #     torch_patience: int          # early stopping patience (epochs w/o val improv)

# # def load_config(path: str) -> CONFIG:
# #     with open(path, "r", encoding="utf-8") as f:
# #         d = json.load(f)
# #     cfg = CONFIG(**d)

# #     # Normalize/validate a couple of fields
# #     cfg.global_feature_mode = str(cfg.global_feature_mode).lower()
# #     assert cfg.global_feature_mode in {"all", "cls", "mean", "none"}, \
# #         "global_feature_mode must be one of {'all','cls','mean','none'}"

# #     # Ensure output dirs exist
# #     os.makedirs(cfg.cache_dir, exist_ok=True)
# #     os.makedirs(cfg.coef_out_dir, exist_ok=True)
# #     os.makedirs(cfg.inference_out_dir, exist_ok=True)
# #     return cfg

# # # =============================
# # # LabelMe reading helpers
# # # =============================

# # @dataclass
# # class Shape:
# #     label: str
# #     shape_type: str
# #     points: List[Tuple[float, float]]

# # def _read_image_from_labelme_data(data: dict, json_dir: str) -> Image.Image:
# #     """
# #     Prefer the embedded `imageData` field (base64). If absent, fallback to imagePath relative to the JSON.
# #     """
# #     image_bytes_b64 = data.get("imageData")
# #     if image_bytes_b64:
# #         # imageData may contain a data URI prefix like: "data:image/png;base64,...."
# #         if isinstance(image_bytes_b64, str) and image_bytes_b64.startswith("data:"):
# #             image_bytes_b64 = image_bytes_b64.split(",", 1)[-1]
# #         img_bytes = base64.b64decode(image_bytes_b64)
# #         return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# #     # Fallback (not expected per the new requirement, but kept for robustness)
# #     image_path = data.get("imagePath", "")
# #     if image_path:
# #         candidate = image_path if os.path.isabs(image_path) else os.path.join(json_dir, image_path)
# #         if os.path.exists(candidate):
# #             return Image.open(candidate).convert("RGB")
# #     raise FileNotFoundError("No embedded imageData and no resolvable imagePath in LabelMe JSON.")

# # def read_labelme_json(json_path: str) -> Tuple[Image.Image, int, int, List[Shape]]:
# #     """Return (PIL.Image, image_width, image_height, shapes). Image is decoded from JSON `imageData`."""
# #     with open(json_path, "r", encoding="utf-8") as f:
# #         data = json.load(f)

# #     img = _read_image_from_labelme_data(data, os.path.dirname(json_path))
# #     w = int(data.get("imageWidth", img.width) or img.width)
# #     h = int(data.get("imageHeight", img.height) or img.height)

# #     shapes: List[Shape] = []
# #     for sh in data.get("shapes", []) or []:
# #         label = (sh.get("label", "") or "").strip()
# #         shape_type = (sh.get("shape_type", "polygon") or "polygon").lower()
# #         raw_points = sh.get("points", []) or []
# #         pts = [(float(x), float(y)) for x, y in raw_points]
# #         if shape_type == "rectangle" and len(pts) == 2:
# #             (x1, y1), (x2, y2) = pts
# #             x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
# #             y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
# #             pts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
# #             shape_type = "polygon"
# #         shapes.append(Shape(label=label, shape_type=shape_type, points=pts))
# #     return img, w, h, shapes

# # # =============================
# # # Rasterization: shapes → pixel mask
# # # =============================

# # def _circle_bbox_from_two_points(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float, float, float]:
# #     cx, cy = p0
# #     px, py = p1
# #     r = math.hypot(px - cx, py - cy)
# #     return (cx - r, cy - r, cx + r, cy + r)

# # def rasterize_mask(
# #     width: int,
# #     height: int,
# #     shapes: List[Shape],
# #     class_map: Dict[str, int],
# #     line_width: int = 1,
# # ) -> np.ndarray:
# #     """
# #     Rasterize LabelMe shapes into an integer mask (H, W) with class ids.

# #     Later shapes overwrite earlier ones if overlapping. Unknown labels are ignored.
# #     Supports: polygon/rectangle, line/linestrip, point, circle (center + point).

# #     NOTE: Background is intentionally NOT part of class_map. We fill background with 255 so it is filtered out later.
# #     """
# #     bg_value = 255  # sentinel that won't match valid class ids
# #     mask_img = Image.new("L", (width, height), color=bg_value)
# #     draw = ImageDraw.Draw(mask_img)

# #     for sh in shapes:
# #         lbl = (sh.label or "").strip().lower()
# #         if lbl not in class_map:
# #             continue
# #         cls_id = int(class_map[lbl])
# #         st = sh.shape_type.lower()
# #         pts = sh.points

# #         if st in {"polygon"} and len(pts) >= 3:
# #             draw.polygon(pts, fill=cls_id)
# #         elif st in {"linestrip", "polyline"} and len(pts) >= 2:
# #             draw.line(pts, fill=cls_id, width=line_width)
# #         elif st == "line" and len(pts) == 2:
# #             draw.line(pts, fill=cls_id, width=line_width)
# #         elif st == "point" and len(pts) == 1:
# #             x, y = pts[0]
# #             draw.point((x, y), fill=cls_id)
# #         elif st == "circle" and len(pts) == 2:
# #             bbox = _circle_bbox_from_two_points(pts[0], pts[1])
# #             draw.ellipse(bbox, fill=cls_id)
# #         elif st == "rectangle" and len(pts) == 2:
# #             (x1, y1), (x2, y2) = pts
# #             x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
# #             y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
# #             draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill=cls_id)
# #         else:
# #             if len(pts) >= 3:
# #                 draw.polygon(pts, fill=cls_id)

# #     return np.array(mask_img, dtype=np.int64)

# # # =============================
# # # DINOv3 feature extraction
# # # =============================

# # @dataclass
# # class DinoOutputs:
# #     patch_tokens: np.ndarray  # (N_patches, D)
# #     global_tokens: np.ndarray # (5, D) — [CLS] + 4 registers
# #     grid_hw: Tuple[int, int]  # (H_patches, W_patches)

# # def get_dino_tokens(
# #     img: Image.Image,
# #     processor: AutoImageProcessor,
# #     model: AutoModel,
# #     target_resolution: int,
# #     patch_size: int,
# #     device: torch.device,
# # ) -> DinoOutputs:
# #     """
# #     Extract DINOv3 tokens: [CLS] + 4 register tokens + patch tokens.
# #     """
# #     target = (target_resolution // patch_size) * patch_size
# #     inputs = processor(images=img, size={"height": target, "width": target}, do_center_crop=False, return_tensors="pt")
# #     inputs = {k: v.to(device) for k, v in inputs.items()}
# #     with torch.inference_mode():
# #         out = model(**inputs)
# #     hs = out.last_hidden_state  # (1, 1+4+N_patches, D)
# #     glob = hs[:, :5, :].squeeze(0).detach().cpu().numpy()     # (5, D)
# #     patch = hs[:, 5:, :].squeeze(0).detach().cpu().numpy()    # (N_patches, D)
# #     n_tokens, D = patch.shape
# #     side = int(round(math.sqrt(n_tokens)))
# #     if side * side != n_tokens:
# #         raise RuntimeError(f"Expected square patch grid; got {n_tokens} tokens")
# #     return DinoOutputs(patch_tokens=patch, global_tokens=glob, grid_hw=(side, side))

# # def build_concat_features(
# #     patch_tokens: np.ndarray,
# #     global_tokens: np.ndarray,
# #     mode: str = "all",
# # ) -> np.ndarray:
# #     """
# #     Concatenate global features with local (per-patch) tokens.

# #     mode:
# #       - "cls":   concat CLS only → [patch, cls]         → dim = 2D
# #       - "mean":  concat mean(global 5) → [patch, mean]  → dim = 2D
# #       - "all":   concat all 5 global tokens flattened   → [patch, glob_flat] → dim = 6D
# #       - "none":  no concat (just patch tokens)          → dim = D

# #     FIX: Properly handle non-"all" modes and add "none".
# #     """
# #     D = patch_tokens.shape[1]
# #     mode = (mode or "all").lower()

# #     if mode == "none":
# #         feats = patch_tokens

# #     elif mode == "cls":
# #         g = global_tokens[0]                    # (D,)
# #         g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
# #         feats = np.concatenate([patch_tokens, g_rep], axis=1)

# #     elif mode == "mean":
# #         g = global_tokens.mean(axis=0)          # (D,)
# #         g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
# #         feats = np.concatenate([patch_tokens, g_rep], axis=1)

# #     else:  # "all"
# #         g = global_tokens.reshape(-1)           # (5D,)
# #         g_rep = np.broadcast_to(g, (patch_tokens.shape[0], 5 * D))
# #         feats = np.concatenate([patch_tokens, g_rep], axis=1)

# #     return feats.astype(np.float32)

# # # =============================
# # # Dataset assembly
# # # =============================

# # def list_labelme_jsons(root: str) -> List[str]:
# #     if not root:
# #         return []
# #     res = []
# #     for dirpath, _, files in os.walk(root):
# #         for fn in files:
# #             if fn.lower().endswith(".json"):
# #                 res.append(os.path.join(dirpath, fn))
# #     res.sort()
# #     return res

# # def cache_key(json_path: str, cfg: CONFIG) -> str:
# #     h = hashlib.sha256()
# #     h.update(json_path.encode("utf-8"))
# #     h.update(cfg.model_id.encode("utf-8"))
# #     h.update(str(cfg.target_resolution).encode("utf-8"))
# #     h.update(str(cfg.patch_size).encode("utf-8"))
# #     # NOTE: cache is for tokens only (independent of global concat mode)
# #     return h.hexdigest()[:16]

# # def extract_features_and_labels(
# #     json_paths: List[str],
# #     cfg: CONFIG,
# #     processor: AutoImageProcessor,
# #     model: AutoModel,
# #     device: torch.device,
# # ) -> Tuple[np.ndarray, np.ndarray]:
# #     X_list: List[np.ndarray] = []
# #     y_list: List[np.ndarray] = []

# #     for i, ann_path in enumerate(json_paths, 1):
# #         if not os.path.exists(ann_path):
# #             print(f"[WARN] Missing annotation: {ann_path}")
# #             continue

# #         img, w, h, shapes = read_labelme_json(ann_path)
# #         mask = rasterize_mask(width=w, height=h, shapes=shapes, class_map=cfg.class_map)

# #         ck = cache_key(ann_path, cfg)
# #         feat_cache = os.path.join(cfg.cache_dir, f"{ck}.npz")

# #         if os.path.exists(feat_cache):
# #             data = np.load(feat_cache)
# #             patch_tokens = data["patch_tokens"]
# #             global_tokens = data["global_tokens"]
# #             gh = int(data["gh"]) if "gh" in data else int(round(math.sqrt(patch_tokens.shape[0])))
# #             gw = int(data["gw"]) if "gw" in data else int(round(math.sqrt(patch_tokens.shape[0])))
# #         else:
# #             out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
# #             patch_tokens = out.patch_tokens
# #             global_tokens = out.global_tokens
# #             gh, gw = out.grid_hw
# #             np.savez_compressed(feat_cache, patch_tokens=patch_tokens, global_tokens=global_tokens, gh=gh, gw=gw)

# #         # Downsample mask to patch grid using nearest neighbor
# #         mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
# #         mask_grid = mask_img.resize((gw, gh), resample=Image.NEAREST)
# #         y = np.array(mask_grid, dtype=np.int64).reshape(-1)

# #         X = build_concat_features(patch_tokens, global_tokens, cfg.global_feature_mode)

# #         X_list.append(X)
# #         y_list.append(y)

# #         if i % 20 == 0:
# #             print(f"Processed {i}/{len(json_paths)}: {os.path.basename(ann_path)}")

# #     if not X_list:
# #         raise RuntimeError("No data loaded — check LabelMe paths.")

# #     X = np.concatenate(X_list, axis=0)
# #     y = np.concatenate(y_list, axis=0)
# #     return X, y

# # # =============================
# # # PyTorch Training / Evaluation
# # # =============================

# # class TorchLinearProbe(nn.Module):
# #     def __init__(self, in_dim: int, n_classes: int):
# #         super().__init__()
# #         self.fc = nn.Linear(in_dim, n_classes)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return self.fc(x)

# # def train_linear_probe_torch(
# #     X_train: np.ndarray,
# #     y_train: np.ndarray,
# #     X_val: Optional[np.ndarray],
# #     y_val: Optional[np.ndarray],
# #     n_classes: int,
# #     cfg: CONFIG,
# #     device: torch.device,
# # ):
# #     in_dim = X_train.shape[1]
# #     model = TorchLinearProbe(in_dim, n_classes).to(device)

# #     # Data
# #     xtr = torch.from_numpy(X_train)
# #     ytr = torch.from_numpy(y_train.astype(np.int64))
# #     train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.torch_batch_size, shuffle=True, drop_last=False)

# #     if X_val is not None and y_val is not None:
# #         xv = torch.from_numpy(X_val)
# #         yv = torch.from_numpy(y_val.astype(np.int64))
# #         val_loader = DataLoader(TensorDataset(xv, yv), batch_size=cfg.torch_batch_size, shuffle=False)
# #     else:
# #         val_loader = None

# #     # Optimizer
# #     opt_name = cfg.torch_optimizer.lower()
# #     if opt_name == "sgd":
# #         opt = torch.optim.SGD(model.parameters(), lr=cfg.torch_lr, momentum=cfg.torch_momentum, weight_decay=cfg.torch_weight_decay)
# #     elif opt_name == "lbfgs":
# #         opt = torch.optim.LBFGS(model.parameters(), lr=cfg.torch_lbfgs_lr, max_iter=100, history_size=50, line_search_fn="strong_wolfe")
# #     else:
# #         opt = torch.optim.Adam(model.parameters(), lr=cfg.torch_lr, weight_decay=cfg.torch_weight_decay)

# #     criterion = nn.CrossEntropyLoss()

# #     best_val = float("inf")
# #     best_state = None
# #     no_improve = 0

# #     def run_epoch(loader, train=True):
# #         model.train(train)
# #         total_loss, total_correct, total_n = 0.0, 0, 0
# #         if isinstance(opt, torch.optim.LBFGS) and train:
# #             def closure():
# #                 opt.zero_grad(set_to_none=True)
# #                 loss_accum = 0.0
# #                 for xb, yb in loader:
# #                     xb = xb.to(device)
# #                     yb = yb.to(device)
# #                     logits = model(xb)
# #                     loss = criterion(logits, yb)
# #                     loss.backward()
# #                     loss_accum += loss.item() * xb.size(0)
# #                 return torch.tensor(loss_accum / len(loader.dataset), device=device, requires_grad=True)
# #             _ = opt.step(closure)
# #             model.eval()
# #             with torch.no_grad():
# #                 for xb, yb in loader:
# #                     xb = xb.to(device)
# #                     yb = yb.to(device)
# #                     logits = model(xb)
# #                     loss = criterion(logits, yb)
# #                     total_loss += loss.item() * xb.size(0)
# #                     pred = logits.argmax(dim=1)
# #                     total_correct += (pred == yb).sum().item()
# #                     total_n += xb.size(0)
# #             return total_loss / total_n, total_correct / total_n
# #         else:
# #             for xb, yb in loader:
# #                 xb = xb.to(device)
# #                 yb = yb.to(device)
# #                 if train:
# #                     opt.zero_grad(set_to_none=True)
# #                 logits = model(xb)
# #                 loss = criterion(logits, yb)
# #                 if train:
# #                     loss.backward()
# #                     opt.step()
# #                 total_loss += loss.item() * xb.size(0)
# #                 pred = logits.argmax(dim=1)
# #                 total_correct += (pred == yb).sum().item()
# #                 total_n += xb.size(0)
# #             return total_loss / total_n, total_correct / total_n

# #     for epoch in range(1, cfg.torch_epochs + 1):
# #         t0 = time.time()
# #         tr_loss, tr_acc = run_epoch(train_loader, train=True)
# #         if val_loader is not None:
# #             model.eval()
# #             with torch.no_grad():
# #                 val_loss, correct_val, n_val = 0.0, 0, 0
# #                 for xb, yb in val_loader:
# #                     xb = xb.to(device)
# #                     yb = yb.to(device)
# #                     logits = model(xb)
# #                     loss = criterion(logits, yb)
# #                     val_loss += loss.item() * xb.size(0)
# #                     correct_val += (logits.argmax(1) == yb).sum().item()
# #                     n_val += xb.size(0)
# #             val_loss /= max(1, n_val)
# #             val_acc = correct_val / max(1, n_val)
# #         else:
# #             val_loss, val_acc = tr_loss, tr_acc

# #         dt = time.time() - t0
# #         print(f"[torch] epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
# #               f"val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.2f}s")

# #         if val_loss + 1e-6 < best_val:
# #             best_val = val_loss
# #             best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
# #             no_improve = 0
# #         else:
# #             no_improve += 1
# #             if no_improve >= cfg.torch_patience:
# #                 print("Early stopping.")
# #                 break

# #     if best_state is not None:
# #         model.load_state_dict(best_state)

# #     return model

# # def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, target_names: List[str], device: torch.device) -> None:
# #     model.eval()
# #     xb = torch.from_numpy(X).to(device)
# #     with torch.no_grad():
# #         logits = model(xb)
# #         pred = logits.argmax(1).cpu().numpy()
# #     acc = float((pred == y).mean())
# #     print(f"Accuracy: {acc:.4f}")
# #     labels = list(range(len(target_names)))
# #     cm = confusion_matrix(y, pred, labels=labels)
# #     print("Confusion matrix (rows=true, cols=pred):")
# #     print(cm)
# #     print(classification_report(y, pred, labels=labels, target_names=target_names, digits=4))

# # def save_linear_probe_torch(model: nn.Module, class_names: List[str], cfg: CONFIG, fname: str = "linear_probe_dinov3_vitl16_torch.pt") -> str:
# #     out_path = os.path.join(cfg.coef_out_dir, fname)
# #     meta = {
# #         "model_id": cfg.model_id,
# #         "target_resolution": cfg.target_resolution,
# #         "patch_size": cfg.patch_size,
# #         "global_feature_mode": cfg.global_feature_mode,
# #         "active_class_names": class_names,
# #         "class_map": cfg.class_map,
# #     }
# #     torch.save({"state_dict": model.state_dict(), "meta": meta}, out_path)
# #     print(f"Saved torch linear probe to: {out_path}")
# #     return out_path

# # # =============================
# # # Inference & Visualization
# # # =============================

# # def predict_patch_grid_torch(
# #     img: Image.Image,
# #     cfg: CONFIG,
# #     processor: AutoImageProcessor,
# #     model: AutoModel,
# #     device: torch.device,
# #     probe: nn.Module,
# # ) -> Tuple[np.ndarray, Tuple[int, int]]:
# #     out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
# #     X = build_concat_features(out.patch_tokens, out.global_tokens, cfg.global_feature_mode)
# #     X = torch.from_numpy(X).to(device)
# #     with torch.no_grad():
# #         logits = probe(X)
# #         y_pred = logits.argmax(1).cpu().numpy()  # values in [0, n_active_classes-1]
# #     H, W = out.grid_hw
# #     return y_pred.reshape(H, W), out.grid_hw

# # def upsample_to_image(mask_grid: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
# #     H, W = target_hw
# #     grid_img = Image.fromarray(mask_grid.astype(np.int32), mode="I")
# #     up = grid_img.resize((W, H), resample=Image.NEAREST)
# #     return np.array(up, dtype=np.int64)

# # def overlay_mask_on_image(
# #     img: Image.Image,
# #     mask: np.ndarray,
# #     alpha: float = 0.5,
# #     class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
# #     show: bool = True,
# #     save_path: Optional[str] = None,
# # ):
# #     base = img.convert("RGBA")
# #     H, W = base.height, base.width

# #     if class_colors is None:
# #         # Default palette indexed by *new* class indices 0..K-1
# #         class_colors = {
# #             0: (0, 114, 178),      # sky (blue-ish)
# #             1: (230, 159, 0),      # cloud (orange)
# #             2: (204, 121, 167),    # contamination (pink)
# #         }

# #     color_mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))
# #     for cls_id, rgb in class_colors.items():
# #         ys, xs = np.where(mask == cls_id)
# #         if ys.size == 0:
# #             continue
# #         layer = Image.new("RGBA", (W, H), (*rgb, int(255 * alpha)))
# #         m = Image.fromarray((mask == cls_id).astype(np.uint8) * 255, mode="L")
# #         color_mask = Image.composite(layer, color_mask, m)

# #     blended = Image.alpha_composite(base, color_mask)

# #     if show:
# #         plt.figure(figsize=(8, 8))
# #         plt.imshow(blended)
# #         plt.axis('off')
# #         plt.tight_layout()
# #         plt.show()

# #     if save_path:
# #         blended.convert("RGB").save(save_path)
# #         print(f"Saved overlay to: {save_path}")

# #     return blended

# # def run_inference_on_images(
# #     json_paths: List[str],
# #     cfg: CONFIG,
# #     processor: AutoImageProcessor,
# #     model: AutoModel,
# #     device: torch.device,
# #     probe: nn.Module,
# #     save_results: bool = True,
# #     show_plots: bool = False,
# #     save_overlays: bool = True,
# # ) -> List[Tuple[str, np.ndarray]]:
# #     results: List[Tuple[str, np.ndarray]] = []
# #     for p in json_paths:
# #         img, w, h, shapes = read_labelme_json(p)
# #         grid_mask, _ = predict_patch_grid_torch(img, cfg, processor, model, device, probe)
# #         up_mask = upsample_to_image(grid_mask, (img.height, img.width))
# #         results.append((p, up_mask))

# #         stem = os.path.splitext(os.path.basename(p))[0]

# #         if save_results:
# #             out_mask_path = os.path.join(cfg.inference_out_dir, f"{stem}_mask.png")
# #             Image.fromarray(up_mask.astype(np.uint8)).save(out_mask_path)
# #             print(f"Saved mask: {out_mask_path}")

# #         if save_overlays:
# #             out_overlay_path = os.path.join(cfg.inference_out_dir, f"{stem}_overlay.jpg")
# #             overlay_mask_on_image(img, up_mask, alpha=0.5, show=show_plots, save_path=out_overlay_path)

# #         if show_plots and not save_overlays:
# #             plt.figure(figsize=(8, 8))
# #             plt.imshow(img)
# #             plt.imshow(up_mask, alpha=0.4)
# #             plt.axis('off')
# #             plt.tight_layout()
# #             plt.show()
# #     return results

# # # =============================
# # # Helpers for label mapping
# # # =============================

# # def _build_label_mapping(cfg: CONFIG) -> Tuple[List[int], Dict[int, int], List[str]]:
# #     """Return (valid_ids, oldid_to_new, class_names_in_new_index_order)."""
# #     valid_names = list(cfg.active_class_names)
# #     valid_ids = [cfg.class_map[n] for n in valid_names]
# #     id_to_new = {old: i for i, old in enumerate(valid_ids)}
# #     return valid_ids, id_to_new, valid_names

# # def _filter_and_remap_labels(y: np.ndarray, valid_ids: List[int], id_to_new: Dict[int, int]):
# #     keep = np.isin(y, valid_ids)
# #     y_kept = y[keep]
# #     y_new = np.vectorize(id_to_new.get)(y_kept)
# #     return keep, y_new.astype(np.int64)

# # # =============================
# # # Main
# # # =============================

# # def main():
# #     parser = argparse.ArgumentParser(description="DINOv3 linear probe on LabelMe (global concat, torch-only)")
# #     parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
# #     args = parser.parse_args()

# #     cfg = load_config(args.config)

# #     torch.manual_seed(cfg.random_state)
# #     np.random.seed(cfg.random_state)

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Using device: {device}")

# #     print("Loading DINOv3 model...")
# #     processor = AutoImageProcessor.from_pretrained(cfg.model_id)
# #     model = AutoModel.from_pretrained(cfg.model_id).to(device)
# #     model.eval()

# #     train_jsons = list_labelme_jsons(cfg.train_labelme_root)
# #     test_jsons  = list_labelme_jsons(cfg.test_labelme_root)
# #     print({"train_jsons": len(train_jsons), "test_jsons": len(test_jsons)})

# #     valid_ids, id_to_new, class_names = _build_label_mapping(cfg)

# #     # ------- Train data
# #     print("Extracting train features...")
# #     X_train_all, y_train_raw_all = extract_features_and_labels(train_jsons, cfg, processor, model, device)
# #     keep_tr, y_train_all = _filter_and_remap_labels(y_train_raw_all, valid_ids, id_to_new)
# #     X_train_all = X_train_all[keep_tr]

# #     # 90/10 train/val split
# #     n = X_train_all.shape[0]
# #     idx = np.arange(n)
# #     np.random.shuffle(idx)
# #     n_val = max(1, int(0.1 * n))
# #     val_idx = idx[:n_val]
# #     tr_idx = idx[n_val:]
# #     X_val, y_val = X_train_all[val_idx], y_train_all[val_idx]
# #     X_tr,  y_tr  = X_train_all[tr_idx],  y_train_all[tr_idx]

# #     n_classes = len(class_names)
# #     probe = train_linear_probe_torch(X_tr, y_tr, X_val, y_val, n_classes=n_classes, cfg=cfg, device=device)
# #     save_linear_probe_torch(probe, class_names, cfg)

# #     # ------- Test data
# #     print("Extracting test features...")
# #     X_test_all, y_test_raw_all = extract_features_and_labels(test_jsons, cfg, processor, model, device)
# #     keep_te, y_test = _filter_and_remap_labels(y_test_raw_all, valid_ids, id_to_new)
# #     X_test = X_test_all[keep_te]

# #     # ------- Eval
# #     evaluate_torch(probe, X_test, y_test, target_names=class_names, device=device)

# # if __name__ == "__main__":
# #     main()
# from __future__ import annotations

# import os
# import io
# import json
# import math
# import time
# import base64
# import hashlib
# import argparse
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional

# import numpy as np
# from PIL import Image, ImageDraw

# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from transformers import AutoImageProcessor, AutoModel
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt

# # =============================
# # Config loading
# # =============================

# @dataclass
# class CONFIG:
#     # Input folders (each contains *.json produced by LabelMe)
#     train_labelme_root: str
#     test_labelme_root: str

#     # Output dirs
#     cache_dir: str
#     coef_out_dir: str
#     inference_out_dir: str

#     # DINOv3
#     model_id: str                # e.g., "facebook/dinov3-vitl16-pretrain-lvd1689m"
#     target_resolution: int       # resized (kept multiple of patch size)
#     patch_size: int              # 16 for ViT-L/16, etc.

#     # Classes — map LabelMe labels → integer ids (original ids in data)
#     class_map: Dict[str, int]

#     # Only consider these classes for training/eval/inference labels
#     active_class_names: List[str]

#     # Global feature concat mode: "all" | "cls" | "mean" | "none"
#     global_feature_mode: str

#     # PyTorch linear probe
#     random_state: int
#     torch_batch_size: int
#     torch_epochs: int
#     torch_optimizer: str         # "adam", "sgd", or "lbfgs"
#     torch_lr: float              # for adam/sgd
#     torch_weight_decay: float
#     torch_momentum: float        # for sgd
#     torch_lbfgs_lr: float        # for lbfgs
#     torch_patience: int          # early stopping patience (epochs w/o val improv)

#     # Per-JSON validation split ratio for training folder
#     per_json_val_ratio: float    # e.g., 0.1


# def load_config(path: str) -> CONFIG:
#     with open(path, "r", encoding="utf-8") as f:
#         d = json.load(f)
#     cfg = CONFIG(**d)

#     # Normalize/validate a couple of fields
#     cfg.global_feature_mode = str(cfg.global_feature_mode).lower()
#     assert cfg.global_feature_mode in {"all", "cls", "mean", "none"}, \
#         "global_feature_mode must be one of {'all','cls','mean','none'}"

#     assert 0.0 < cfg.per_json_val_ratio < 0.5, "per_json_val_ratio should be in (0, 0.5) for a sensible split."

#     # Ensure output dirs exist
#     os.makedirs(cfg.cache_dir, exist_ok=True)
#     os.makedirs(cfg.coef_out_dir, exist_ok=True)
#     os.makedirs(cfg.inference_out_dir, exist_ok=True)
#     return cfg

# # =============================
# # LabelMe reading helpers
# # =============================

# @dataclass
# class Shape:
#     label: str
#     shape_type: str
#     points: List[Tuple[float, float]]

# def _read_image_from_labelme_data(data: dict, json_dir: str) -> Image.Image:
#     """
#     Prefer the embedded `imageData` field (base64). If absent, fallback to imagePath relative to the JSON.
#     """
#     image_bytes_b64 = data.get("imageData")
#     if image_bytes_b64:
#         # imageData may contain a data URI prefix like: "data:image/png;base64,...."
#         if isinstance(image_bytes_b64, str) and image_bytes_b64.startswith("data:"):
#             image_bytes_b64 = image_bytes_b64.split(",", 1)[-1]
#         img_bytes = base64.b64decode(image_bytes_b64)
#         return Image.open(io.BytesIO(img_bytes)).convert("RGB")

#     # Fallback (kept for robustness)
#     image_path = data.get("imagePath", "")
#     if image_path:
#         candidate = image_path if os.path.isabs(image_path) else os.path.join(json_dir, image_path)
#         if os.path.exists(candidate):
#             return Image.open(candidate).convert("RGB")
#     raise FileNotFoundError("No embedded imageData and no resolvable imagePath in LabelMe JSON.")

# def read_labelme_json(json_path: str) -> Tuple[Image.Image, int, int, List[Shape]]:
#     """Return (PIL.Image, image_width, image_height, shapes). Image is decoded from JSON `imageData`."""
#     with open(json_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     img = _read_image_from_labelme_data(data, os.path.dirname(json_path))
#     w = int(data.get("imageWidth", img.width) or img.width)
#     h = int(data.get("imageHeight", img.height) or img.height)

#     shapes: List[Shape] = []
#     for sh in data.get("shapes", []) or []:
#         label = (sh.get("label", "") or "").strip()
#         shape_type = (sh.get("shape_type", "polygon") or "polygon").lower()
#         raw_points = sh.get("points", []) or []
#         pts = [(float(x), float(y)) for x, y in raw_points]
#         if shape_type == "rectangle" and len(pts) == 2:
#             (x1, y1), (x2, y2) = pts
#             x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
#             y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
#             pts = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
#             shape_type = "polygon"
#         shapes.append(Shape(label=label, shape_type=shape_type, points=pts))
#     return img, w, h, shapes

# # =============================
# # Rasterization: shapes → pixel mask
# # =============================

# def _circle_bbox_from_two_points(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float, float, float]:
#     cx, cy = p0
#     px, py = p1
#     r = math.hypot(px - cx, py - cy)
#     return (cx - r, cy - r, cx + r, cy + r)

# def rasterize_mask(
#     width: int,
#     height: int,
#     shapes: List[Shape],
#     class_map: Dict[str, int],
#     line_width: int = 1,
# ) -> np.ndarray:
#     """
#     Rasterize LabelMe shapes into an integer mask (H, W) with class ids.

#     Later shapes overwrite earlier ones if overlapping. Unknown labels are ignored.
#     Supports: polygon/rectangle, line/linestrip, point, circle (center + point).

#     NOTE: Background is intentionally NOT part of class_map. We fill background with 255 so it is filtered out later.
#     """
#     bg_value = 255  # sentinel that won't match valid class ids
#     mask_img = Image.new("L", (width, height), color=bg_value)
#     draw = ImageDraw.Draw(mask_img)

#     for sh in shapes:
#         lbl = (sh.label or "").strip().lower()
#         if lbl not in class_map:
#             continue
#         cls_id = int(class_map[lbl])
#         st = sh.shape_type.lower()
#         pts = sh.points

#         if st in {"polygon"} and len(pts) >= 3:
#             draw.polygon(pts, fill=cls_id)
#         elif st in {"linestrip", "polyline"} and len(pts) >= 2:
#             draw.line(pts, fill=cls_id, width=line_width)
#         elif st == "line" and len(pts) == 2:
#             draw.line(pts, fill=cls_id, width=line_width)
#         elif st == "point" and len(pts) == 1:
#             x, y = pts[0]
#             draw.point((x, y), fill=cls_id)
#         elif st == "circle" and len(pts) == 2:
#             bbox = _circle_bbox_from_two_points(pts[0], pts[1])
#             draw.ellipse(bbox, fill=cls_id)
#         elif st == "rectangle" and len(pts) == 2:
#             (x1, y1), (x2, y2) = pts
#             x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
#             y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)
#             draw.polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], fill=cls_id)
#         else:
#             if len(pts) >= 3:
#                 draw.polygon(pts, fill=cls_id)

#     return np.array(mask_img, dtype=np.int64)

# # =============================
# # DINOv3 feature extraction
# # =============================

# @dataclass
# class DinoOutputs:
#     patch_tokens: np.ndarray  # (N_patches, D)
#     global_tokens: np.ndarray # (5, D) — [CLS] + 4 registers
#     grid_hw: Tuple[int, int]  # (H_patches, W_patches)

# def get_dino_tokens(
#     img: Image.Image,
#     processor: AutoImageProcessor,
#     model: AutoModel,
#     target_resolution: int,
#     patch_size: int,
#     device: torch.device,
# ) -> DinoOutputs:
#     """
#     Extract DINOv3 tokens: [CLS] + 4 register tokens + patch tokens.
#     """
#     target = (target_resolution // patch_size) * patch_size
#     inputs = processor(images=img, size={"height": target, "width": target}, do_center_crop=False, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     with torch.inference_mode():
#         out = model(**inputs)
#     hs = out.last_hidden_state  # (1, 1+4+N_patches, D)
#     glob = hs[:, :5, :].squeeze(0).detach().cpu().numpy()     # (5, D)
#     patch = hs[:, 5:, :].squeeze(0).detach().cpu().numpy()    # (N_patches, D)
#     n_tokens, _ = patch.shape
#     side = int(round(math.sqrt(n_tokens)))
#     if side * side != n_tokens:
#         raise RuntimeError(f"Expected square patch grid; got {n_tokens} tokens")
#     return DinoOutputs(patch_tokens=patch, global_tokens=glob, grid_hw=(side, side))

# def build_concat_features(
#     patch_tokens: np.ndarray,
#     global_tokens: np.ndarray,
#     mode: str = "all",
# ) -> np.ndarray:
#     """
#     Concatenate global features with local (per-patch) tokens.

#     mode:
#       - "cls":   concat CLS only → [patch, cls]         → dim = 2D
#       - "mean":  concat mean(global 5) → [patch, mean]  → dim = 2D
#       - "all":   concat all 5 global tokens flattened   → [patch, glob_flat] → dim = 6D
#       - "none":  no concat (just patch tokens)          → dim = D
#     """
#     D = patch_tokens.shape[1]
#     mode = (mode or "all").lower()

#     if mode == "none":
#         feats = patch_tokens

#     elif mode == "cls":
#         g = global_tokens[0]                    # (D,)
#         g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
#         feats = np.concatenate([patch_tokens, g_rep], axis=1)

#     elif mode == "mean":
#         g = global_tokens.mean(axis=0)          # (D,)
#         g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
#         feats = np.concatenate([patch_tokens, g_rep], axis=1)

#     else:  # "all"
#         g = global_tokens.reshape(-1)           # (5D,)
#         g_rep = np.broadcast_to(g, (patch_tokens.shape[0], 5 * D))
#         feats = np.concatenate([patch_tokens, g_rep], axis=1)

#     return feats.astype(np.float32)

# # =============================
# # Dataset assembly
# # =============================

# def list_labelme_jsons(root: str) -> List[str]:
#     if not root:
#         return []
#     res = []
#     for dirpath, _, files in os.walk(root):
#         for fn in files:
#             if fn.lower().endswith(".json"):
#                 res.append(os.path.join(dirpath, fn))
#     res.sort()
#     return res

# def cache_key(json_path: str, cfg: CONFIG) -> str:
#     h = hashlib.sha256()
#     h.update(json_path.encode("utf-8"))
#     h.update(cfg.model_id.encode("utf-8"))
#     h.update(str(cfg.target_resolution).encode("utf-8"))
#     h.update(str(cfg.patch_size).encode("utf-8"))
#     # Cache stores tokens only (concat mode is applied later)
#     return h.hexdigest()[:16]

# def extract_tokens_for_json(
#     json_path: str,
#     cfg: CONFIG,
#     processor: AutoImageProcessor,
#     model: AutoModel,
#     device: torch.device,
# ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int], Image.Image, np.ndarray]:
#     """
#     For a single JSON: returns (patch_tokens, global_tokens, (gh,gw), img, mask)
#     """
#     img, w, h, shapes = read_labelme_json(json_path)
#     mask = rasterize_mask(width=w, height=h, shapes=shapes, class_map=cfg.class_map)

#     ck = cache_key(json_path, cfg)
#     feat_cache = os.path.join(cfg.cache_dir, f"{ck}.npz")

#     if os.path.exists(feat_cache):
#         data = np.load(feat_cache)
#         patch_tokens = data["patch_tokens"]
#         global_tokens = data["global_tokens"]
#         gh = int(data["gh"]) if "gh" in data else int(round(math.sqrt(patch_tokens.shape[0])))
#         gw = int(data["gw"]) if "gw" in data else int(round(math.sqrt(patch_tokens.shape[0])))
#     else:
#         out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
#         patch_tokens = out.patch_tokens
#         global_tokens = out.global_tokens
#         gh, gw = out.grid_hw
#         np.savez_compressed(feat_cache, patch_tokens=patch_tokens, global_tokens=global_tokens, gh=gh, gw=gw)

#     return patch_tokens, global_tokens, (gh, gw), img, mask

# def mask_to_patch_labels(mask: np.ndarray, grid_hw: Tuple[int, int]) -> np.ndarray:
#     gh, gw = grid_hw
#     mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
#     mask_grid = mask_img.resize((gw, gh), resample=Image.NEAREST)
#     y = np.array(mask_grid, dtype=np.int64).reshape(-1)
#     return y

# # =============================
# # PyTorch Training / Evaluation
# # =============================

# class TorchLinearProbe(nn.Module):
#     def __init__(self, in_dim: int, n_classes: int):
#         super().__init__()
#         self.fc = nn.Linear(in_dim, n_classes)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.fc(x)

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
#     train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.torch_batch_size, shuffle=True, drop_last=False)

#     if X_val is not None and y_val is not None:
#         xv = torch.from_numpy(X_val)
#         yv = torch.from_numpy(y_val.astype(np.int64))
#         val_loader = DataLoader(TensorDataset(xv, yv), batch_size=cfg.torch_batch_size, shuffle=False)
#     else:
#         val_loader = None

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
#                     xb = xb.to(device)
#                     yb = yb.to(device)
#                     logits = model(xb)
#                     loss = criterion(logits, yb)
#                     total_loss += loss.item() * xb.size(0)
#                     pred = logits.argmax(dim=1)
#                     total_correct += (pred == yb).sum().item()
#                     total_n += xb.size(0)
#             return total_loss / total_n, total_correct / total_n
#         else:
#             for xb, yb in loader:
#                 xb = xb.to(device)
#                 yb = yb.to(device)
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
#             return total_loss / total_n, total_correct / total_n

#     for epoch in range(1, cfg.torch_epochs + 1):
#         t0 = time.time()
#         tr_loss, tr_acc = run_epoch(train_loader, train=True)
#         if val_loader is not None:
#             model.eval()
#             with torch.no_grad():
#                 val_loss, correct_val, n_val = 0.0, 0, 0
#                 for xb, yb in val_loader:
#                     xb = xb.to(device)
#                     yb = yb.to(device)
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

# def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, target_names: List[str], device: torch.device) -> None:
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
#     print(classification_report(y, pred, labels=labels, target_names=target_names, digits=4))

# def save_linear_probe_torch(model: nn.Module, class_names: List[str], cfg: CONFIG, fname: str = "linear_probe_dinov3_vitl16_torch.pt") -> str:
#     out_path = os.path.join(cfg.coef_out_dir, fname)
#     meta = {
#         "model_id": cfg.model_id,
#         "target_resolution": cfg.target_resolution,
#         "patch_size": cfg.patch_size,
#         "global_feature_mode": cfg.global_feature_mode,
#         "active_class_names": class_names,
#         "class_map": cfg.class_map,
#     }
#     torch.save({"state_dict": model.state_dict(), "meta": meta}, out_path)
#     print(f"Saved torch linear probe to: {out_path}")
#     return out_path

# # =============================
# # Inference & Visualization
# # =============================

# def predict_patch_grid_torch(
#     img: Image.Image,
#     cfg: CONFIG,
#     processor: AutoImageProcessor,
#     model: AutoModel,
#     device: torch.device,
#     probe: nn.Module,
# ) -> Tuple[np.ndarray, Tuple[int, int]]:
#     out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
#     X = build_concat_features(out.patch_tokens, out.global_tokens, cfg.global_feature_mode)
#     X = torch.from_numpy(X).to(device)
#     with torch.no_grad():
#         logits = probe(X)
#         y_pred = logits.argmax(1).cpu().numpy()  # values in [0, n_active_classes-1]
#     H, W = out.grid_hw
#     return y_pred.reshape(H, W), out.grid_hw

# def upsample_to_image(mask_grid: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
#     H, W = target_hw
#     grid_img = Image.fromarray(mask_grid.astype(np.int32), mode="I")
#     up = grid_img.resize((W, H), resample=Image.NEAREST)
#     return np.array(up, dtype=np.int64)

# def overlay_mask_on_image(
#     img: Image.Image,
#     mask: np.ndarray,
#     alpha: float = 0.5,
#     class_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
#     show: bool = True,
#     save_path: Optional[str] = None,
# ):
#     base = img.convert("RGBA")
#     H, W = base.height, base.width

#     if class_colors is None:
#         # Default palette indexed by *new* class indices 0..K-1
#         class_colors = {
#             0: (0, 114, 178),      # sky (blue-ish)
#             1: (230, 159, 0),      # cloud (orange)
#             2: (204, 121, 167),    # contamination (pink)
#         }

#     color_mask = Image.new("RGBA", (W, H), (0, 0, 0, 0))
#     for cls_id, rgb in class_colors.items():
#         ys, xs = np.where(mask == cls_id)
#         if ys.size == 0:
#             continue
#         layer = Image.new("RGBA", (W, H), (*rgb, int(255 * alpha)))
#         m = Image.fromarray((mask == cls_id).astype(np.uint8) * 255, mode="L")
#         color_mask = Image.composite(layer, color_mask, m)

#     blended = Image.alpha_composite(base, color_mask)

#     if show:
#         plt.figure(figsize=(8, 8))
#         plt.imshow(blended)
#         plt.axis('off')
#         plt.tight_layout()
#         plt.show()

#     if save_path:
#         blended.convert("RGB").save(save_path)
#         print(f"Saved overlay to: {save_path}")

#     return blended

# # =============================
# # Label mapping helpers
# # =============================

# def _build_label_mapping(cfg: CONFIG) -> Tuple[List[int], Dict[int, int], List[str]]:
#     """Return (valid_ids, oldid_to_new, class_names_in_new_index_order)."""
#     valid_names = list(cfg.active_class_names)
#     valid_ids = [cfg.class_map[n] for n in valid_names]
#     id_to_new = {old: i for i, old in enumerate(valid_ids)}
#     return valid_ids, id_to_new, valid_names

# def _filter_and_remap_labels(y: np.ndarray, valid_ids: List[int], id_to_new: Dict[int, int]):
#     keep = np.isin(y, valid_ids)
#     y_kept = y[keep]
#     y_new = np.vectorize(id_to_new.get)(y_kept)
#     return keep, y_new.astype(np.int64)

# # =============================
# # Main (with per-JSON train/val split)
# # =============================

# def main():
#     parser = argparse.ArgumentParser(description="DINOv3 linear probe on LabelMe (global concat, torch-only)")
#     parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
#     args = parser.parse_args()

#     cfg = load_config(args.config)

#     torch.manual_seed(cfg.random_state)
#     np.random.seed(cfg.random_state)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     print("Loading DINOv3 model...")
#     processor = AutoImageProcessor.from_pretrained(cfg.model_id)
#     model = AutoModel.from_pretrained(cfg.model_id).to(device)
#     model.eval()

#     train_jsons = list_labelme_jsons(cfg.train_labelme_root)
#     test_jsons  = list_labelme_jsons(cfg.test_labelme_root)
#     print({"train_jsons": len(train_jsons), "test_jsons": len(test_jsons)})

#     valid_ids, id_to_new, class_names = _build_label_mapping(cfg)

#     # ------- Build train/val from EACH train JSON (per-JSON split)
#     X_tr_list, y_tr_list = [], []
#     X_val_list, y_val_list = [], []

#     for i, jp in enumerate(train_jsons, 1):
#         patch_tokens, global_tokens, (gh, gw), img, mask = extract_tokens_for_json(jp, cfg, processor, model, device)
#         y_full = mask_to_patch_labels(mask, (gh, gw))
#         X_full = build_concat_features(patch_tokens, global_tokens, cfg.global_feature_mode)

#         # Filter and remap per JSON
#         keep, y_mapped = _filter_and_remap_labels(y_full, valid_ids, id_to_new)
#         if keep.sum() == 0:
#             continue
#         X_kept = X_full[keep]

#         # Per-JSON random split into train/val
#         n_kept = X_kept.shape[0]
#         idx = np.arange(n_kept)
#         np.random.shuffle(idx)
#         n_val = max(1, int(cfg.per_json_val_ratio * n_kept))
#         val_idx = idx[:n_val]
#         tr_idx  = idx[n_val:]

#         X_tr_list.append(X_kept[tr_idx])
#         y_tr_list.append(y_mapped[tr_idx])
#         X_val_list.append(X_kept[val_idx])
#         y_val_list.append(y_mapped[val_idx])

#         if i % 20 == 0:
#             print(f"[train json] processed {i}/{len(train_jsons)}")

#     if not X_tr_list:
#         raise RuntimeError("No training data assembled. Check train_labelme_root and class_map/active_class_names.")

#     X_tr = np.concatenate(X_tr_list, axis=0)
#     y_tr = np.concatenate(y_tr_list, axis=0)
#     X_val = np.concatenate(X_val_list, axis=0)
#     y_val = np.concatenate(y_val_list, axis=0)

#     print(f"[assembled] train_patches={X_tr.shape[0]}, val_patches={X_val.shape[0]}")

#     # ------- Train
#     n_classes = len(class_names)
#     probe = train_linear_probe_torch(X_tr, y_tr, X_val, y_val, n_classes=n_classes, cfg=cfg, device=device)
#     save_linear_probe_torch(probe, class_names, cfg)

#     # ------- Build test set (entirely from test JSONs; no split inside test)
#     X_te_list, y_te_list = [], []
#     for i, jp in enumerate(test_jsons, 1):
#         patch_tokens, global_tokens, (gh, gw), img, mask = extract_tokens_for_json(jp, cfg, processor, model, device)
#         y_full = mask_to_patch_labels(mask, (gh, gw))
#         X_full = build_concat_features(patch_tokens, global_tokens, cfg.global_feature_mode)

#         keep, y_mapped = _filter_and_remap_labels(y_full, valid_ids, id_to_new)
#         if keep.sum() == 0:
#             continue
#         X_kept = X_full[keep]

#         X_te_list.append(X_kept)
#         y_te_list.append(y_mapped)

#         if i % 20 == 0:
#             print(f"[test json] processed {i}/{len(test_jsons)}")

#     if not X_te_list:
#         raise RuntimeError("No test data assembled. Check test_labelme_root and class_map/active_class_names.")

#     X_test = np.concatenate(X_te_list, axis=0)
#     y_test = np.concatenate(y_te_list, axis=0)

#     # ------- Eval
#     print("Evaluating on test patches ...")
#     evaluate_torch(probe, X_test, y_test, target_names=class_names, device=device)

# if __name__ == "__main__":
#     main()
from __future__ import annotations

import os
import io
import json
import math
import time
import base64
import hashlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# =============================
# Config loading
# =============================

@dataclass
class CONFIG:
    # Input folders (each contains *.json produced by LabelMe)
    train_labelme_root: str
    test_labelme_root: str
    val_labelme_root: str
    # Output dirs
    cache_dir: str
    coef_out_dir: str
    inference_out_dir: str

    # DINOv3
    model_id: str                # e.g., "facebook/dinov3-vitl16-pretrain-lvd1689m"
    target_resolution: int       # resized (kept multiple of patch size)
    patch_size: int              # 16 for ViT-L/16, etc.

    # Classes — map LabelMe labels → integer ids (original ids in data)
    class_map: Dict[str, int]

    # Only consider these classes for training/eval/inference labels
    active_class_names: List[str]

    # Global feature concat mode: "all" | "cls" | "mean" | "none"
    global_feature_mode: str

    # PyTorch linear probe
    random_state: int
    torch_batch_size: int
    torch_epochs: int
    torch_optimizer: str         # "adam", "sgd", or "lbfgs"
    torch_lr: float              # for adam/sgd
    torch_weight_decay: float
    torch_momentum: float        # for sgd
    torch_lbfgs_lr: float        # for lbfgs
    torch_patience: int          # early stopping patience (epochs w/o val improv)

    # JSON-level validation split ratio (split the file list first)
    json_val_ratio: float        # e.g., 0.1

def load_config(path: str) -> CONFIG:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    cfg = CONFIG(**d)

    # Normalize/validate a couple of fields
    cfg.global_feature_mode = str(cfg.global_feature_mode).lower()
    assert cfg.global_feature_mode in {"all", "cls", "mean", "none"}, \
        "global_feature_mode must be one of {'all','cls','mean','none'}"

    assert 0.0 < cfg.json_val_ratio < 0.5, "json_val_ratio should be in (0, 0.5)."

    # Ensure output dirs exist
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(cfg.coef_out_dir, exist_ok=True)
    os.makedirs(cfg.inference_out_dir, exist_ok=True)
    return cfg

# =============================
# LabelMe reading helpers
# =============================

@dataclass
class Shape:
    label: str
    shape_type: str
    points: List[Tuple[float, float]]

def _read_image_from_labelme_data(data: dict, json_dir: str) -> Image.Image:
    """
    Prefer the embedded `imageData` field (base64). If absent, fallback to imagePath relative to the JSON.
    """
    image_bytes_b64 = data.get("imageData")
    if image_bytes_b64:
        # imageData may contain a data URI prefix like: "data:image/png;base64,...."
        if isinstance(image_bytes_b64, str) and image_bytes_b64.startswith("data:"):
            image_bytes_b64 = image_bytes_b64.split(",", 1)[-1]
        img_bytes = base64.b64decode(image_bytes_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Fallback (kept for robustness)
    image_path = data.get("imagePath", "")
    if image_path:
        candidate = image_path if os.path.isabs(image_path) else os.path.join(json_dir, image_path)
        if os.path.exists(candidate):
            return Image.open(candidate).convert("RGB")
    raise FileNotFoundError("No embedded imageData and no resolvable imagePath in LabelMe JSON.")

def read_labelme_json(json_path: str) -> Tuple[Image.Image, int, int, List[Shape]]:
    """Return (PIL.Image, image_width, image_height, shapes). Image is decoded from JSON `imageData`."""
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

# =============================
# Rasterization: shapes → pixel mask
# =============================

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
    """
    Rasterize LabelMe shapes into an integer mask (H, W) with class ids.

    Later shapes overwrite earlier ones if overlapping. Unknown labels are ignored.
    Supports: polygon/rectangle, line/linestrip, point, circle (center + point).

    NOTE: Background is intentionally NOT part of class_map. We fill background with 255 so it is filtered out later.
    """
    bg_value = 255  # sentinel that won't match valid class ids
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

# =============================
# DINOv3 feature extraction
# =============================

@dataclass
class DinoOutputs:
    patch_tokens: np.ndarray  # (N_patches, D)
    global_tokens: np.ndarray # (5, D) — [CLS] + 4 registers
    grid_hw: Tuple[int, int]  # (H_patches, W_patches)

def get_dino_tokens(
    img: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModel,
    target_resolution: int,
    patch_size: int,
    device: torch.device,
) -> DinoOutputs:
    """
    Extract DINOv3 tokens: [CLS] + 4 register tokens + patch tokens.
    """
    target = (target_resolution // patch_size) * patch_size
    inputs = processor(images=img, size={"height": target, "width": target}, do_center_crop=False, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    hs = out.last_hidden_state  # (1, 1+4+N_patches, D)
    glob = hs[:, :5, :].squeeze(0).detach().cpu().numpy()     # (5, D)
    patch = hs[:, 5:, :].squeeze(0).detach().cpu().numpy()    # (N_patches, D)
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
    """
    Concatenate global features with local (per-patch) tokens.

    mode:
      - "cls":   concat CLS only → [patch, cls]         → dim = 2D
      - "mean":  concat mean(global 5) → [patch, mean]  → dim = 2D
      - "all":   concat all 5 global tokens flattened   → [patch, glob_flat] → dim = 6D
      - "none":  no concat (just patch tokens)          → dim = D

    (Fix) non-"all" modes now correctly tile a D-length vector; "none" returns patch tokens only.
    """
    D = patch_tokens.shape[1]
    mode = (mode or "all").lower()

    if mode == "none":
        feats = patch_tokens

    elif mode == "cls":
        g = global_tokens[0]                    # (D,)
        g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)

    elif mode == "mean":
        g = global_tokens.mean(axis=0)          # (D,)
        g_rep = np.broadcast_to(g, (patch_tokens.shape[0], D))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)

    else:  # "all"
        g = global_tokens.reshape(-1)           # (5D,)
        g_rep = np.broadcast_to(g, (patch_tokens.shape[0], 5 * D))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)

    return feats.astype(np.float32)

# =============================
# Dataset assembly
# =============================

def list_labelme_jsons(root: str) -> List[str]:
    if not root:
        return []
    res = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(".json"):
                res.append(os.path.join(dirpath, fn))
    res.sort()
    return res

def cache_key(json_path: str, cfg: CONFIG) -> str:
    h = hashlib.sha256()
    h.update(json_path.encode("utf-8"))
    h.update(cfg.model_id.encode("utf-8"))
    h.update(str(cfg.target_resolution).encode("utf-8"))
    h.update(str(cfg.patch_size).encode("utf-8"))
    # Cache stores tokens only (concat mode is applied later)
    return h.hexdigest()[:16]

def extract_features_and_labels_for_jsons(
    json_paths: List[str],
    cfg: CONFIG,
    processor: AutoImageProcessor,
    model: AutoModel,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a list of JSON files, extract features and labels (concatenated across files).
    """
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for i, jp in enumerate(json_paths, 1):
        # Decode image & shapes and rasterize mask
        img, w, h, shapes = read_labelme_json(jp)
        mask = rasterize_mask(width=w, height=h, shapes=shapes, class_map=cfg.class_map)

        # Load or compute DINO tokens
        ck = cache_key(jp, cfg)
        feat_cache = os.path.join(cfg.cache_dir, f"{ck}.npz")

        if os.path.exists(feat_cache):
            data = np.load(feat_cache)
            patch_tokens = data["patch_tokens"]
            global_tokens = data["global_tokens"]
            gh = int(data["gh"]) if "gh" in data else int(round(math.sqrt(patch_tokens.shape[0])))
            gw = int(data["gw"]) if "gw" in data else int(round(math.sqrt(patch_tokens.shape[0])))
        else:
            out = get_dino_tokens(img, processor, model, cfg.target_resolution, cfg.patch_size, device)
            patch_tokens = out.patch_tokens
            global_tokens = out.global_tokens
            gh, gw = out.grid_hw
            np.savez_compressed(feat_cache, patch_tokens=patch_tokens, global_tokens=global_tokens, gh=gh, gw=gw)

        # Downsample mask to patch grid using nearest neighbor
        mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
        mask_grid = mask_img.resize((gw, gh), resample=Image.NEAREST)
        y_full = np.array(mask_grid, dtype=np.int64).reshape(-1)

        # Build features with the selected global feature mode
        X_full = build_concat_features(patch_tokens, global_tokens, cfg.global_feature_mode)

        # Filter & remap labels to consecutive indices 0..K-1
        valid_ids, id_to_new, _ = _build_label_mapping(cfg)
        keep = np.isin(y_full, valid_ids)
        if not keep.any():
            continue
        y_kept = np.vectorize(id_to_new.get)(y_full[keep]).astype(np.int64)
        X_kept = X_full[keep]

        X_list.append(X_kept)
        y_list.append(y_kept)

        if i % 20 == 0:
            print(f"[extract] processed {i}/{len(json_paths)}")

    if not X_list:
        raise RuntimeError("No data extracted; check paths and class_map/active_class_names.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

# =============================
# PyTorch Training / Evaluation
# =============================

class TorchLinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

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

    # Data
    xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train.astype(np.int64))
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=cfg.torch_batch_size, shuffle=False, drop_last=False,num_workers=10)

    val_loader = None
    if X_val is not None and y_val is not None and len(X_val) > 0:
        xv = torch.from_numpy(X_val)
        yv = torch.from_numpy(y_val.astype(np.int64))
        val_loader = DataLoader(TensorDataset(xv, yv), batch_size=cfg.torch_batch_size, shuffle=False,num_workers=10)

    # Optimizer
    opt_name = cfg.torch_optimizer.lower()
    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.torch_lr, momentum=cfg.torch_momentum, weight_decay=cfg.torch_weight_decay)
    elif opt_name == "lbfgs":
        opt = torch.optim.LBFGS(model.parameters(), lr=cfg.torch_lbfgs_lr, max_iter=100, history_size=50, line_search_fn="strong_wolfe")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.torch_lr, weight_decay=cfg.torch_weight_decay)

    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    def run_epoch(loader, train=True):
        model.train(train)
        total_loss, total_correct, total_n = 0.0, 0, 0
        if isinstance(opt, torch.optim.LBFGS) and train:
            def closure():
                opt.zero_grad(set_to_none=True)
                loss_accum = 0.0
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    loss_accum += loss.item() * xb.size(0)
                return torch.tensor(loss_accum / len(loader.dataset), device=device, requires_grad=True)
            _ = opt.step(closure)
            model.eval()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    total_loss += loss.item() * xb.size(0)
                    pred = logits.argmax(dim=1)
                    total_correct += (pred == yb).sum().item()
                    total_n += xb.size(0)
            return total_loss / total_n, total_correct / total_n
        else:
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                if train:
                    opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                if train:
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * xb.size(0)
                pred = logits.argmax(dim=1)
                total_correct += (pred == yb).sum().item()
                total_n += xb.size(0)
            return total_loss / total_n, total_correct / total_n

    for epoch in range(1, cfg.torch_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss, correct_val, n_val = 0.0, 0, 0
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
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
        print(f"[torch] epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
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

def evaluate_torch(model: nn.Module, X: np.ndarray, y: np.ndarray, target_names: List[str], device: torch.device) -> None:
    model.eval()
    xb = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
    acc = float((pred == y).mean())
    print(f"Accuracy: {acc:.4f}")
    labels = list(range(len(target_names)))
    cm = confusion_matrix(y, pred, labels=labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    results = classification_report(y, pred, labels=labels, target_names=target_names, digits=4, output_dict = True)
    print(type(results))
    print(results)
    return acc,results

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

# =============================
# Label mapping helpers
# =============================

def _build_label_mapping(cfg: CONFIG) -> Tuple[List[int], Dict[int, int], List[str]]:
    """Return (valid_ids, oldid_to_new, class_names_in_new_index_order)."""
    valid_names = list(cfg.active_class_names)
    valid_ids = [cfg.class_map[n] for n in valid_names]
    id_to_new = {old: i for i, old in enumerate(valid_ids)}
    return valid_ids, id_to_new, valid_names

# =============================
# Utilities
# =============================

def split_jsons(json_paths: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """
    Deterministically split a list of JSON files into train and val lists.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(json_paths))
    n_val = max(1, int(round(val_ratio * len(json_paths))))
    val_idx = set(perm[:n_val].tolist())
    train_list, val_list = [], []
    for i, p in enumerate(json_paths):
        (val_list if i in val_idx else train_list).append(p)
    return train_list, val_list

# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser(description="DINOv3 linear probe on LabelMe (global concat, torch-only)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    torch.manual_seed(cfg.random_state)
    np.random.seed(cfg.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading DINOv3 model...")
    processor = AutoImageProcessor.from_pretrained(cfg.model_id)
    model = AutoModel.from_pretrained(cfg.model_id).to(device)
    model.eval()

    # List JSONs
    train_jsons = list_labelme_jsons(cfg.train_labelme_root)
    test_jsons      = list_labelme_jsons(cfg.test_labelme_root)
    val_jsons       = list_labelme_jsons(cfg.val_labelme_root)
    print({"train_jsons": len(train_jsons), "test_jsons": len(test_jsons), "val_jsons": len(val_jsons)})

    # Build consistent label mapping up front
    valid_ids, id_to_new, class_names = _build_label_mapping(cfg)

    # JSON-level split (no patch mixing between train and val)
    #train_jsons, val_jsons = split_jsons(all_train_jsons, cfg.json_val_ratio, seed=cfg.random_state)
    print({"train_jsons": len(train_jsons), "val_jsons": len(val_jsons)})

    # ------- Extract features for each split
    print("Extracting TRAIN features...")
    X_tr, y_tr = extract_features_and_labels_for_jsons(train_jsons, cfg, processor, model, device)

    print("Extracting VAL features...")
    X_val, y_val = extract_features_and_labels_for_jsons(val_jsons, cfg, processor, model, device)

    print(f"[assembled] train_patches={X_tr.shape[0]}, val_patches={X_val.shape[0]}")

    # ------- Train
    n_classes = len(class_names)
    probe = train_linear_probe_torch(X_tr, y_tr, X_val, y_val, n_classes=n_classes, cfg=cfg, device=device)
    save_linear_probe_torch(probe, class_names, cfg)

    # ------- Test (pure hold-out folder)
    print("Extracting TEST features...")
    X_test, y_test = extract_features_and_labels_for_jsons(test_jsons, cfg, processor, model, device)

    # ------- Eval
    print("Evaluating on test patches ...")
    total_accuracy, results = evaluate_torch(probe, X_test, y_test, target_names=class_names, device=device)
    result_dict = {"overall_accuracy": total_accuracy, "per_class_results": results}
    # save to json file
    with open(os.path.join(cfg.inference_out_dir,f"test_results_seed_{cfg.random_state}.json"), "w") as f:
        json.dump(result_dict, f)
if __name__ == "__main__":
    main()
