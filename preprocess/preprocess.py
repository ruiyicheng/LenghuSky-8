"""
Image preprocessing utilities for batch cropping, resizing, optional intensity scaling,
and deterministic timestamp-based renaming of photos.

Overview
--------
This module provides:

1) `chop_then_resize_to_jpg(...)`
   - Loads a JPEG (respecting EXIF orientation) or Canon RAW (.CR2 via `rawpy`),
   - (Optionally) canonicalizes orientation so the long side is horizontal,
   - Crops using *fractional* windows expressed in a stable logical frame,
   - Resizes to a target resolution,
   - (Optionally) applies intensity scaling,
   - Saves the result as a high-quality JPEG.

2) `preprocess_dir(...)`
   - Recursively processes a directory of images,
   - Derives an output filename from a timestamp parsed from the source filename
     (with robust patterns) or from EXIF `DateTimeOriginal` (JPEG fallback),
   - Deduplicates by appending numeric suffixes when timestamps collide.

Key Concepts
------------
• Logical crop frame:
  After canonicalization (if enabled), the short side is *height* (H) and the long side is *width* (W).
  Fractions are specified as:
    - `x_frac=(x0, x1)` along the SHORT side (height),
    - `y_frac=(y0, y1)` along the LONG side (width).
  Fractions are clamped to [0,1] and must have positive extent.

• Orientation handling:
  - `canonicalize_long_to_width=True` rotates portrait images 90° CCW so the long side becomes width.
  - `restore_original=True` rotates back to the source orientation after processing.

• Intensity scaling (optional):
  - `scale=None | "minmax" | "z3"`, with `scale_mode="global" | "per_channel"`.
  - "minmax": linear map [min..max] → [0..255].
  - "z3": linear map [(μ−1σ)..(μ+3σ)] → [0..255], values outside are clipped.
  - `linear_scale=True` (legacy) aliases `scale="minmax"` if `scale` is None.

• Formats:
  - Inputs: `.jpg`, `.jpeg`, and `.cr2` (case-insensitive).
  - Output: JPEG (quality=95, optimize, progressive, 4:2:0 subsampling).
  - Transparency (if present) is flattened against white before saving.

Timestamp & Naming
------------------
- Filenames are parsed for timestamps using several robust patterns (e.g., `YYYY_MM_DD_hh_mm_ss`,
  `YYYYMMDDhhmmss`, or tolerant sequences of digits).
- For JPEGs without a parsable name, EXIF `DateTimeOriginal` is used if available.
- Output filenames use the format: `YYYY-MM-DD-hh-mm-ss.jpg`.
- If multiple inputs resolve to the same timestamp, suffixes `_1`, `_2`, ... are appended.

Dependencies
------------
Required:
  - Pillow (`PIL`)
Optional (used when features require them):
  - `rawpy` (to read `.CR2`)
  - `numpy` (when `scale` is not `None`)

Examples
--------
Process a single file:
    chop_then_resize_to_jpg(
        input_path="input.jpg",
        output_path="out.jpg",
        x_frac=(0.0, 1.0),
        y_frac=(0.166667, 0.83333),
        target_size=(512, 512),
        scale="z3",
        scale_mode="global",
        canonicalize_long_to_width=True,
        restore_original=False,
    )

Batch a directory (see __main__ for a concrete invocation):
    preprocess_dir(
        input_dir="path/to/input_dir",
        output_dir="path/to/output_dir",
        x_frac=(0.0, 1.0),
        y_frac=(0.166667, 0.83333),
        target_size=(512, 512),
        overwrite=False,
        recurse=True,
        canonicalize_long_to_width=True,
        restore_original=False,
        scale="z3",
        scale_mode="global",
    )

Caveats
-------
- Fractions are clamped; zero-extent crops raise `ValueError`.
- `scale` must be one of `{None, "minmax", "z3"}`; `scale_mode` must be `{ "global", "per_channel" }`.
- `.CR2` support requires `rawpy`; scaling requires `numpy`.
- Canonicalization rotates portrait images CCW; restoration rotates CW if enabled.

"""

from pathlib import Path
import re
from datetime import datetime
from collections import Counter

def chop_then_resize_to_jpg(
    input_path,
    output_path,
    x_frac=(0.0, 1.0),
    y_frac=(0.0, 1.0),
    target_size=(1024, 1024),
    *,
    # Back-compat alias for minmax scaling:
    linear_scale=True,
    # Unified scaling selector:
    scale: str | None = "z3",      # None | "minmax" | "z3"
    scale_mode: str = "global",    # "global" | "per_channel"
    # NEW: orientation controls
    canonicalize_long_to_width: bool = True,  # rotate portrait -> landscape before chopping
    restore_original: bool = False,           # rotate back to source orientation at the end
):
    """
    Step 0 (NEW): CANONICALIZE orientation.
        If enabled and the image is portrait (H > W), rotate 90° CCW so that the
        long side is always horizontal (width). This guarantees a single "normal"
        frame for cropping, eliminating the 90° mismatch across outputs.

    Step 1: CHOP using fractional windows after EXIF orientation (and after
        canonicalization if enabled).
        IMPORTANT: Fractions are specified in a logical frame where:
          - x runs along the SHORT side,
          - y runs along the LONG side.
        After canonicalization, the short side is height and the long side is width,
        so we can use one consistent mapping.

    Step 2: RESIZE to target_size.
    Step 3 (optional): SCALE to [0,255].
        - scale="minmax": linear stretch min..max -> 0..255
        - scale="z3":     linear stretch (μ-3σ)..(μ+3σ) -> 0..255 (clip outside)
        - scale_mode:     "global" or "per_channel"
    Step 4: (optional) RESTORE original orientation.
    Step 5: Save JPEG.

    Supports: .jpg/.jpeg and .CR2 (via rawpy)
    """
    from PIL import Image, ImageOps

    # Back-compat
    if scale is None and linear_scale:
        scale = "minmax"
    if scale not in (None, "minmax", "z3"):
        raise ValueError("scale must be None, 'minmax', or 'z3'.")
    if scale_mode not in ("global", "per_channel"):
        raise ValueError("scale_mode must be 'global' or 'per_channel'.")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = input_path.suffix.lower()
    is_cr2 = ext == ".cr2"

    # --- Load image (respect EXIF for JPEGs) ---
    if is_cr2:
        try:
            import rawpy
        except ImportError as e:
            raise RuntimeError("CR2 support requires 'rawpy' (pip install rawpy).") from e
        with rawpy.imread(str(input_path)) as raw:
            print("[INFO] Processing CR2 image...")
            rgb8 = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                output_bps=8,
            )
        img = Image.fromarray(rgb8, mode="RGB")
    else:
        img = Image.open(str(input_path))
        img = ImageOps.exif_transpose(img)  # normalize EXIF orientation to pixels
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

    # --- Validate and clamp fractions ---
    def _validate_pair(p, name):
        if not (isinstance(p, (tuple, list)) and len(p) == 2):
            raise ValueError(f"{name} must be a (begin, end) pair.")
        a, b = float(p[0]), float(p[1])
        if a > b:
            raise ValueError(f"{name}[0] must be <= {name}[1]. Got {p}.")
        a = max(0.0, min(1.0, a))
        b = max(0.0, min(1.0, b))
        if a == b:
            raise ValueError(f"{name} has zero extent after clamping: {p}.")
        return a, b

    x0f, x1f = _validate_pair(x_frac, "x_frac")  # along SHORT side
    y0f, y1f = _validate_pair(y_frac, "y_frac")  # along LONG  side

    # --- Step 0: CANONICALIZE orientation (portrait -> landscape) ---
    # Track if we rotated so we can optionally restore later.
    was_portrait = False
    if canonicalize_long_to_width:
        W0, H0 = img.size
        if H0 > W0:
            was_portrait = True
            # Rotate 90° CCW to make long side horizontal consistently
            img = img.rotate(90, expand=True)  # CCW
            # (If you prefer CW, use img = img.rotate(-90, expand=True))

    # --- Step 1: CHOP in a consistent frame ---
    # After canonicalization (or if already landscape), we have:
    #   short side -> height (H), long side -> width (W)
    W, H = img.size
    # Map logical (x_short, y_long) -> actual (left,right,top,bottom)
    left   = max(0, min(int(round(y0f * W)), W - 1))
    right  = max(left + 1, min(int(round(y1f * W)), W))
    top    = max(0, min(int(round(x0f * H)), H - 1))
    bottom = max(top + 1, min(int(round(x1f * H)), H))
    img = img.crop((left, top, right, bottom))

    # --- Step 2: RESIZE ---
    tgt_w, tgt_h = map(int, target_size)
    if tgt_w <= 0 or tgt_h <= 0:
        raise ValueError("target_size must be positive integers.")
    img = img.resize((tgt_w, tgt_h), Image.Resampling.LANCZOS)

    # Flatten transparency if any (perform before scaling)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # --- Step 3: Optional scaling ---
    if scale is not None:
        import numpy as np
        arr = np.asarray(img).astype(np.float32)  # (H, W, 3)

        if scale_mode == "per_channel":
            if scale == "minmax":
                lo = arr.min(axis=(0, 1), keepdims=True)
                hi = arr.max(axis=(0, 1), keepdims=True)
            else:  # "z3"
                mu = arr.mean(axis=(0, 1), keepdims=True)
                sd = arr.std(axis=(0, 1), keepdims=True)
                # Symmetric 3σ window (matches docstring)
                lo = mu - 1.0 * sd
                hi = mu + 3.0 * sd
                # If you prefer the old asymmetric window, change to: lo = mu - 1.0*sd
        else:  # global
            if scale == "minmax":
                lo_v = float(arr.min())
                hi_v = float(arr.max())
                lo = np.array([[[lo_v, lo_v, lo_v]]], dtype=np.float32)
                hi = np.array([[[hi_v, hi_v, hi_v]]], dtype=np.float32)
            else:  # "z3"
                mu_v = float(arr.mean())
                sd_v = float(arr.std())
                lo_v = mu_v - 1.0 * sd_v
                hi_v = mu_v + 3.0 * sd_v
                lo = np.array([[[lo_v, lo_v, lo_v]]], dtype=np.float32)
                hi = np.array([[[hi_v, hi_v, hi_v]]], dtype=np.float32)

        denom = np.maximum(hi - lo, 1e-6)
        norm = (arr - lo) / denom
        norm = np.clip(norm, 0.0, 1.0)
        arr_out = (norm * 255.0).round().astype(np.uint8)
        img = Image.fromarray(arr_out, mode="RGB")

    # --- Step 4: (optional) RESTORE original orientation ---
    if restore_original and canonicalize_long_to_width and was_portrait:
        # We rotated CCW earlier; rotate CW now to restore.
        img = img.rotate(-90, expand=True)

    # --- Step 5: Save JPEG ---
    img.save(
        str(output_path),
        format="JPEG",
        quality=95,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )


# --------- timestamp parsing helpers ---------
_TS_PATTERNS = [
    # e.g. 2022_04_20_14_42_31, 2022-04-20 14-42-31, 2018-08-09 00^%55^%29
    re.compile(r'(?P<Y>\d{4})\D+(?P<M>\d{2})\D+(?P<D>\d{2})\D+(?P<h>\d{2})\D*(?P<m>\d{2})\D*(?P<s>\d{2})'),
    # e.g. 20220420144231 (compact 14 digits)
    re.compile(r'(?P<Y>\d{4})(?P<M>\d{2})(?P<D>\d{2})(?P<h>\d{2})(?P<m>\d{2})(?P<s>\d{2})'),
]

def _try_parse_from_name(name: str) -> datetime | None:
    """Try to parse a timestamp from a filename (without extension)."""
    for pat in _TS_PATTERNS:
        m = pat.search(name)
        if m:
            try:
                Y = int(m.group('Y')); M = int(m.group('M')); D = int(m.group('D'))
                h = int(m.group('h')); m_ = int(m.group('m')); s = int(m.group('s'))
                return datetime(Y, M, D, h, m_, s)
            except ValueError:
                print(f"[WARN] Invalid timestamp in filename '{name}'.")
    # Fallback: scan for 6 adjacent numeric groups with lengths 4,2,2,2,2,2
    groups = re.findall(r'\d+', name)
    for i in range(len(groups) - 5):
        lens = list(map(len, groups[i:i+6]))
        if lens[0] == 4 and all(L in (1,2) for L in lens[1:]):  # tolerate single-digit M/D/h/m/s
            try:
                Y = int(groups[i+0])
                M = int(groups[i+1].zfill(2))
                D = int(groups[i+2].zfill(2))
                h = int(groups[i+3].zfill(2))
                m_ = int(groups[i+4].zfill(2))
                s = int(groups[i+5].zfill(2))
                return datetime(Y, M, D, h, m_, s)
            except ValueError:
                print(f"[WARN] Invalid timestamp in filename '{name}'.")
    return None

def _try_exif_datetime_jpeg(path: Path) -> datetime | None:
    """Try EXIF DateTimeOriginal for JPEGs only (no extra deps)."""
    try:
        from PIL import Image, ExifTags
        with Image.open(str(path)) as im:
            exif = im.getexif() or {}
        # Find the EXIF tag for DateTimeOriginal (usually 36867)
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        dto = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        # EXIF format: "YYYY:MM:DD HH:MM:SS"
        if isinstance(dto, str) and re.match(r'\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}', dto):
            return datetime.strptime(dto, "%Y:%m:%d %H:%M:%S")
    except Exception:
        print(f"[WARN] Failed to read EXIF from JPEG '{path.name}'.")
    return None

def _timestamp_for_file(path: Path) -> datetime | None:
    stem = path.stem  # filename without extension
    ts = _try_parse_from_name(stem)
    if ts:
        return ts
    # Fallback for JPEGs: EXIF
    if path.suffix.lower() in (".jpg", ".jpeg"):
        return _try_exif_datetime_jpeg(path)
    return None

def _fmt_outname(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"


def preprocess_dir(
    input_dir,
    output_dir,
    x_frac=(0.0, 1.0),
    y_frac=(0.166667, 0.83333),
    target_size=(512, 512),
    *,
    overwrite=False,
    recurse=True,
    # expose the new orientation flags
    canonicalize_long_to_width: bool = True,
    restore_original: bool = False,
    # pass-through for scaling options if needed
    scale: str | None = "z3",
    scale_mode: str = "global",
):
    """
    Recursively preprocess images in `input_dir` and write JPEGs into `output_dir`.

    - Output filename format: yyyy-mm-dd-hh-mm-ss.jpg
    - Supported inputs: .jpg/.jpeg and .CR2 (case-insensitive)
    - Timestamp is parsed from filename; for JPEGs, falls back to EXIF DateTimeOriginal.
    - If multiple files resolve to the same timestamp, suffixes (_1, _2, ...) are appended.
    - Canonicalizes orientation by default so outputs share a consistent (landscape) baseline.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".cr2"}
    files = (input_dir.rglob("*") if recurse else input_dir.glob("*"))
    files = [p for p in files if p.is_file() and p.suffix.lower() in exts]

    used = Counter()
    processed, skipped = 0, 0

    for src in files:
        ts = _timestamp_for_file(src)
        if not ts:
            skipped += 1
            continue

        base = _fmt_outname(ts)
        if used[base]:
            stem = base[:-4]
            base = f"{stem}_{used[stem + '.jpg']}.jpg"
        used[base] += 1

        dst = output_dir / base

        if dst.exists() and not overwrite:
            skipped += 1
            continue

        try:
            chop_then_resize_to_jpg(
                input_path=src,
                output_path=dst,
                x_frac=x_frac,
                y_frac=y_frac,
                target_size=target_size,
                scale=scale,
                scale_mode=scale_mode,
                canonicalize_long_to_width=canonicalize_long_to_width,
                restore_original=restore_original,
            )
            processed += 1
        except Exception as e:
            skipped += 1
            print(f"[WARN] Failed on {src}: {e}")

    print(f"Done. Processed: {processed}, Skipped: {skipped}, Output: {output_dir}")


if __name__ == "__main__":
    preprocess_dir(
        input_dir="F:\\cloud_data\\data\\2018",
        output_dir="D:\\project\\dino\\cloud\\data\\2018",
        x_frac=(0.0, 1.0),
        y_frac=(0.166667, 0.83333),
        target_size=(512, 512),
        canonicalize_long_to_width=True,  # ensure consistent (landscape) frame
        restore_original=False,           # keep canonical orientation in outputs
    )
