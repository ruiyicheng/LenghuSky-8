"""
Inference script for DINOv3 linear probe trained on LabelMe-derived patch labels.

Updates:
- Supports global+local feature concatenation (CLS/MEAN/ALL) to match training.
- Torch .pt probe may include meta: {model_id, target_resolution, patch_size, global_feature_mode, active_class_names}.
- Overlay and "layout image" are based on ARGMAX (the predicted class), not logits.
- Directory flag renamed to --input-dir (was --dir).

Outputs (optional):
  a) Logits artifacts (--logits-out):
     - NPZ with logits on patch grid and upsampled to image size.
     - Per-class grayscale heatmaps (upsampled) as 8-bit PNGs.
  b) Layout & overlays (--overlay-out):
     - layout_mask.png: argmax class map (uint8).
     - layout_color.png: colorized argmax map.
     - overlay.jpg: argmax-colored layout blended over the original.

Usage examples:
    python dinov3_infer_labelme_probe.py \
        --probe path/to/linear_probe_dinov3_vitl16_torch.pt \
        --image path/to/img.jpg \
        --logits-out logs/logits \
        --overlay-out logs/overlay

    python dinov3_infer_labelme_probe.py \
        --probe path/to/linear_probe_dinov3_vitl16_sklearn.npz \
        --input-dir path/to/images \
        --exts jpg png \
        --logits-out logs/logits \
        --overlay-out logs/overlay
"""
from __future__ import annotations

import os
import argparse
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

# ----------------------
# Defaults / Colors
# ----------------------

# If the probe (torch) was trained without background, expect classes like:
#   active_class_names = ["sky", "cloud", "contamination"]
# For sklearn, class_names/classes come from the .npz.
DEFAULT_CLASS_COLORS = {
    # You can override via --class-colors if needed, but these are sensible defaults:
    # If your model is 3-class (no background), indices 0,1,2 map to these colors:
    0: (0, 114, 178),   # "sky" (blue-ish)
    1: (230, 159, 0),   # "cloud" (orange)
    2: (204, 121, 167), # "contamination" (pink)
    # If 4-class (w/ background), add:
    # 3: (0, 0, 0),       # background
}

# ----------------------
# DINO feature extraction (global + patch)
# ----------------------

def load_backbone(model_id: str, device: torch.device):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model

def get_dino_tokens(
    img: Image.Image,
    processor: AutoImageProcessor,
    model: AutoModel,
    target_resolution: int,
    patch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Returns:
      patch_tokens: (N_patches, D)
      global_tokens: (5, D)  # [CLS] + 4 registers
      grid_hw: (Hp, Wp)
    """
    target = (target_resolution // patch_size) * patch_size
    inputs = processor(images=img, size={"height": target, "width": target}, do_center_crop=False, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
    hs = out.last_hidden_state  # (1, 1+4+N, D)
    glob = hs[:, :5, :].squeeze(0).detach().cpu().numpy()     # (5, D)
    patch = hs[:, 5:, :].squeeze(0).detach().cpu().numpy()    # (N, D)
    n_tokens, _ = patch.shape
    side = int(math.sqrt(n_tokens))
    assert side * side == n_tokens, f"Expected square grid, got {n_tokens}"
    return patch, glob, (side, side)

def build_concat_features(patch_tokens: np.ndarray, global_tokens: np.ndarray, mode: str = "all") -> np.ndarray:
    """
    Concatenate global features with local (per-patch) tokens.

    mode:
      - "cls":  concat CLS only → [patch, cls]
      - "mean": concat mean(global 5) → [patch, mean]
      - "all":  concat all 5 global tokens flattened → [patch, glob_flat]
    """
    if mode == "cls":
        g = global_tokens[0]  # (D,)
        g_rep = np.tile(g, (patch_tokens.shape[0], 1))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)  # (N, 2D)
    elif mode == "mean":
        g = global_tokens.mean(axis=0)
        g_rep = np.tile(g, (patch_tokens.shape[0], 1))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)  # (N, 2D)
    else:  # "all"
        g = global_tokens.reshape(-1)  # (5D,)
        g_rep = np.tile(g, (patch_tokens.shape[0], 1))
        feats = np.concatenate([patch_tokens, g_rep], axis=1)  # (N, 6D)
    return feats.astype(np.float32)

# ----------------------
# Probe loaders (sklearn or torch)
# ----------------------

class TorchLinearProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

def load_probe(probe_path: str, device: torch.device):
    """
    Return a dict with keys:
      backend: "torch" | "sklearn"
      classes: np.ndarray of class ids (indices)
      class_names: list[str]
      model_id, target_resolution, patch_size
      global_feature_mode: "all"|"cls"|"mean" (if not present, default to "all")
      forward: callable (x: torch.Tensor[N,D]) -> torch.Tensor[N,C]
      in_dim, n_classes
    """
    meta = {
        "backend": None,
        "classes": None,
        "class_names": None,
        "model_id": None,
        "target_resolution": 1024,
        "patch_size": 16,
        "global_feature_mode": "all",
    }

    ext = os.path.splitext(probe_path)[1].lower()
    if ext == ".npz":
        data = np.load(probe_path, allow_pickle=True)
        coef = data["coef_"]           # (C, D)
        intercept = data["intercept_"] # (C,)
        classes = data["classes_"]
        # Note: sklearn probes from older scripts won't have global_feature_mode or active_class_names.
        class_names = data["class_names"] if "class_names" in data else np.array([str(i) for i in classes], dtype=object)
        model_id = data["model_id"].item() if "model_id" in data else "facebook/dinov3-vitl16-pretrain-lvd1689m"
        target_resolution = int(data["target_resolution"]) if "target_resolution" in data else 1024
        patch_size = int(data["patch_size"]) if "patch_size" in data else 16

        meta.update({
            "backend": "sklearn",
            "classes": classes,
            "class_names": [str(x) for x in (class_names.tolist() if hasattr(class_names, "tolist") else class_names)],
            "model_id": model_id,
            "target_resolution": target_resolution,
            "patch_size": patch_size,
            "global_feature_mode": "all",  # best guess for old files
        })

        coef_t = torch.from_numpy(coef).to(device)  # (C,D)
        b_t = torch.from_numpy(intercept).to(device) # (C,)
        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            return x @ coef_t.t() + b_t
        meta["forward"] = forward_fn
        meta["in_dim"] = coef.shape[1]
        meta["n_classes"] = coef.shape[0]
        return meta

    else:  # assume torch .pt
        payload = torch.load(probe_path, map_location=device)
        state_dict = payload.get("state_dict", payload)
        meta_blob = payload.get("meta", {})

        model_id = meta_blob.get("model_id", "facebook/dinov3-vitl16-pretrain-lvd1689m")
        target_resolution = int(meta_blob.get("target_resolution", 1024))
        patch_size = int(meta_blob.get("patch_size", 16))
        global_feature_mode = meta_blob.get("global_feature_mode", "all")

        # If saved with active_class_names (new training), use them; otherwise fallback to map or indices
        active_names = meta_blob.get("active_class_names")
        class_map = meta_blob.get("class_map")

        W = state_dict["fc.weight"]  # (C, D)
        b = state_dict["fc.bias"]    # (C,)
        n_classes, in_dim = W.shape

        probe = TorchLinearProbe(in_dim, n_classes).to(device)
        probe.load_state_dict(state_dict)
        probe.eval()

        if isinstance(active_names, (list, tuple)) and len(active_names) == n_classes:
            class_names = list(active_names)
            classes = np.arange(n_classes)
        elif isinstance(class_map, dict):
            # sort by class id to align names (older format)
            items = sorted(class_map.items(), key=lambda kv: kv[1])
            class_names = [k for k, _ in items]
            classes = np.array([v for _, v in items])
        else:
            class_names = [str(i) for i in range(n_classes)]
            classes = np.arange(n_classes)

        def forward_fn(x: torch.Tensor) -> torch.Tensor:
            return probe(x)

        meta.update({
            "backend": "torch",
            "classes": classes,
            "class_names": class_names,
            "model_id": model_id,
            "target_resolution": target_resolution,
            "patch_size": patch_size,
            "global_feature_mode": global_feature_mode,
            "forward": forward_fn,
            "in_dim": in_dim,
            "n_classes": n_classes,
        })
        return meta

# ----------------------
# Resizing & visualization
# ----------------------

def upsample_nn(arr_grid: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor upsample for (H_p,W_p,...) -> (H,W,...) numpy arrays."""
    H, W = out_hw
    if arr_grid.ndim == 2:
        im = Image.fromarray(arr_grid.astype(np.float32), mode="F")
        up = im.resize((W, H), resample=Image.NEAREST)
        return np.array(up, dtype=arr_grid.dtype)
    else:
        Hp, Wp, C = arr_grid.shape
        out = np.zeros((H, W, C), dtype=arr_grid.dtype)
        for c in range(C):
            im = Image.fromarray(arr_grid[..., c].astype(np.float32), mode="F")
            up = im.resize((W, H), resample=Image.NEAREST)
            out[..., c] = np.array(up, dtype=arr_grid.dtype)
        return out

def colorize_argmax(mask: np.ndarray, index_to_color: Dict[int, Tuple[int,int,int]]) -> Image.Image:
    """Map discrete argmax mask (H,W) to RGB image using index_to_color dict keyed by class index (0..C-1)."""
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for idx, color in index_to_color.items():
        rgb[mask == idx] = color
    return Image.fromarray(rgb, mode="RGB")

def overlay_on_image(base_img: Image.Image, overlay_img: Image.Image, alpha: float = 0.5) -> Image.Image:
    base = base_img.convert("RGBA")
    ov = overlay_img.convert("RGBA")
    ov.putalpha(int(255 * alpha))
    return Image.alpha_composite(base, ov).convert("RGB")

# ----------------------
# Core inference
# ----------------------

def infer_single_image(
    image_path: str,
    probe_meta: dict,
    device: torch.device,
    logits_out_dir: Optional[str] = None,
    overlay_out_dir: Optional[str] = None,
    class_colors: Optional[Dict[int, Tuple[int,int,int]]] = None,
    save_patch_grid_pngs: bool = True,
    save_upsampled_pngs: bool = True,
    overlay_alpha: float = 0.5,
) -> dict:
    """
    Run inference on one image. Returns a dict of produced paths.

    Saves (if requested):
      - NPZ: patch_grid_logits (H_p,W_p,C), upsampled_logits (H,W,C), classes, class_names
      - Per-class PNGs (grayscale) for logits (upsampled)
      - layout_mask.png (argmax), layout_color.png (colorized argmax), overlay.jpg (argmax-colored overlay)
    """
    out_paths = {}
    classes = probe_meta["classes"]
    class_names = probe_meta["class_names"]
    class_colors = class_colors or DEFAULT_CLASS_COLORS

    # Backbone settings
    model_id = probe_meta.get("model_id") or "facebook/dinov3-vitl16-pretrain-lvd1689m"
    target_res = int(probe_meta.get("target_resolution", 1024))
    patch_size = int(probe_meta.get("patch_size", 16))
    global_mode = probe_meta.get("global_feature_mode", "all")

    # Load backbone
    processor, backbone = load_backbone(model_id, device)

    # Read image
    img = Image.open(image_path).convert("RGB")
    H, W = img.height, img.width

    # Tokens -> features -> logits
    patch_tok, global_tok, (Hp, Wp) = get_dino_tokens(img, processor, backbone, target_res, patch_size, device)
    feats = build_concat_features(patch_tok, global_tok, global_mode)  # (N, D')
    X = torch.from_numpy(feats).to(device)
    with torch.inference_mode():
        logits = probe_meta["forward"](X).detach().cpu().numpy()  # (N, C)
    C = logits.shape[1]
    logits_grid = logits.reshape(Hp, Wp, C)

    # Softmax (patch grid) -> upsample
    probs_grid = np.exp(logits_grid - logits_grid.max(axis=-1, keepdims=True))
    probs_grid /= np.clip(probs_grid.sum(axis=-1, keepdims=True), 1e-12, None)
    logits_up = upsample_nn(logits_grid, (H, W))   # (H,W,C)
    probs_up  = upsample_nn(probs_grid, (H, W))    # (H,W,C)

    # Argmax layout (TYPE MAP)
    argmax_grid = logits_grid.argmax(axis=-1)      # (Hp, Wp)
    layout_mask = upsample_nn(argmax_grid, (H, W)).astype(np.uint8)  # (H, W)

    # Save logits artifacts
    stem = os.path.splitext(os.path.basename(image_path))[0]
    if logits_out_dir:
        os.makedirs(logits_out_dir, exist_ok=True)
        npz_path = os.path.join(logits_out_dir, f"{stem}_logits.npz")
        np.savez_compressed(
            npz_path,
            logits_grid=logits_grid,
            logits_up=logits_up,
            classes=np.array(classes),
            class_names=np.array(class_names, dtype=object),
        )
        out_paths["npz"] = npz_path

        # # Per-class PNGs (upsampled logits → 0..255 per-class normalization)
        # for i, cname in enumerate(class_names):
        #     m = logits_up[..., i]
        #     vmin, vmax = float(m.min()), float(m.max())
        #     denom = (vmax - vmin) if vmax > vmin else 1.0
        #     norm = ((m - vmin) / denom * 255.0).astype(np.uint8)
        #     png_path = os.path.join(logits_out_dir, f"{stem}_logit_{cname}.png")
        #     Image.fromarray(norm, mode="L").save(png_path)

    # Layout and overlay (ARGMAX-BASED)
    if overlay_out_dir:
        os.makedirs(overlay_out_dir, exist_ok=True)

        # Color mapping expects index->color (index = 0..C-1). If you prefer mapping by 'classes' ids,
        # build a mapping {index: class_colors[classes[index]]}:
        index_to_color = {i: class_colors.get(int(classes[i]), (255, 255, 255)) for i in range(C)}

        # Save raw layout mask
        layout_mask_path = os.path.join(overlay_out_dir, f"{stem}_layout_mask.png")
        Image.fromarray(layout_mask, mode="L").save(layout_mask_path)
        out_paths["layout_mask"] = layout_mask_path

        # Save colorized layout
        layout_color_img = colorize_argmax(layout_mask, index_to_color)
        layout_color_path = os.path.join(overlay_out_dir, f"{stem}_layout_color.png")
        layout_color_img.save(layout_color_path)
        out_paths["layout_color"] = layout_color_path

        # Save overlay (argmax-colored)
        overlay_img = overlay_on_image(img, layout_color_img, alpha=overlay_alpha)
        overlay_path = os.path.join(overlay_out_dir, f"{stem}_overlay.jpg")
        overlay_img.save(overlay_path, quality=95)
        out_paths["overlay"] = overlay_path

    return out_paths

# ----------------------
# Directory processing
# ----------------------

def list_images(dir_path: str, exts: List[str]) -> List[str]:
    exts_l = {e.lower().lstrip('.') for e in exts}
    outs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = f.split('.')[-1].lower()
            if ext in exts_l:
                outs.append(os.path.join(root, f))
    outs.sort()
    return outs

def infer_directory(
    dir_path: str,
    probe_meta: dict,
    device: torch.device,
    logits_out_dir: Optional[str] = None,
    overlay_out_dir: Optional[str] = None,
    class_colors: Optional[Dict[int, Tuple[int,int,int]]] = None,
    exts: Optional[List[str]] = None,
) -> List[dict]:
    exts = exts or ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
    paths = list_images(dir_path, exts)
    results = []
    for p in paths:
        print(f"Processing {p} ...")
        res = infer_single_image(
            image_path=p,
            probe_meta=probe_meta,
            device=device,
            logits_out_dir=logits_out_dir,
            overlay_out_dir=overlay_out_dir,
            class_colors=class_colors,
        )
        res["image"] = p
        results.append(res)
    return results

# ----------------------
# CLI
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="DINOv3 linear-probe inference on images")
    parser.add_argument("--probe", default="D:\\project\\dino\\cloud\\log\\experiment1\\linear_probe_dinov3_vitl16_torch.pt", help="Path to saved probe (.npz from sklearn or .pt from torch)")

    in_grp = parser.add_mutually_exclusive_group(required=False)
    in_grp.add_argument("--image", help="Path to a single image")
    in_grp.add_argument("--input-dir", default="D:\\project\\dino\\cloud\\data\\2020", dest="indir", help="Directory of images to process recursively")

    parser.add_argument("--logits-out", default="D:\\project\\dino\\cloud\\log\\experiment1\\inference\\logits", dest="logits_out", help="Directory to save logits artifacts (npz + per-class heatmaps)")
    parser.add_argument("--overlay-out", default="D:\\project\\dino\\cloud\\log\\experiment1\\inference\\overlay", dest="overlay_out", help="Directory to save layout/overlay images")
    parser.add_argument("--exts", nargs="*", default=["jpg", "jpeg", "png"], help="Extensions to scan when using --input-dir")

    parser.add_argument("--model-id", default=None, help="Override model id if probe lacks meta")
    parser.add_argument("--target-res", type=int, default=None, help="Override target resolution if probe lacks meta")
    parser.add_argument("--patch-size", type=int, default=None, help="Override patch size if probe lacks meta")
    parser.add_argument("--global-mode", default=None, choices=["all", "cls", "mean"],
                        help="Override global feature concat mode if probe lacks meta")

    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha (0..1)")

    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    probe_meta = load_probe(args.probe, device)

    # Optional overrides
    if args.model_id:
        probe_meta["model_id"] = args.model_id
    if args.target_res is not None:
        probe_meta["target_resolution"] = int(args.target_res)
    if args.patch_size is not None:
        probe_meta["patch_size"] = int(args.patch_size)
    if args.global_mode is not None:
        probe_meta["global_feature_mode"] = args.global_mode

    if args.image:
        res = infer_single_image(
            image_path=args.image,
            probe_meta=probe_meta,
            device=device,
            logits_out_dir=args.logits_out,
            overlay_out_dir=args.overlay_out,
            class_colors=DEFAULT_CLASS_COLORS,
            overlay_alpha=args.alpha,
        )
        print(res)
    else:
        results = infer_directory(
            dir_path=args.indir,
            probe_meta=probe_meta,
            device=device,
            logits_out_dir=args.logits_out,
            overlay_out_dir=args.overlay_out,
            class_colors=DEFAULT_CLASS_COLORS,
            exts=args.exts,
        )
        print(f"Processed {len(results)} images.")

if __name__ == "__main__":
    main()
