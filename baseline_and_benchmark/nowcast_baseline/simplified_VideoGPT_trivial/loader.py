"""
Loads sequences of logits images and corresponding masks for training/evaluation.
Normalization is needed for the logits images to adapt to VideoGPT
This normalization would not affect the results of trivial mapping baseline.
"""
import os
import re
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from datetime import datetime, timedelta
import cv2

# ---------- utilities ----------

_TIME_PATTERNS = [
    (re.compile(r'(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})'), "%Y-%m-%d-%H-%M-%S"),
    (re.compile(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})'), "%Y_%m_%d_%H_%M_%S"),
    (re.compile(r'(\d{8}_\d{6})'), "%Y%m%d_%H%M%S"),
]

def parse_time_from_string(s: str) -> datetime:
    s = s.strip()
    # support "YYYY-mm-dd HH:MM:SS" and "YYYY-mm-dd-HH-MM-SS"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d-%H-%M-%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized time format: {s}")

def parse_time_from_filename(fname: str) -> datetime:
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    for pat, fmt in _TIME_PATTERNS:
        m = pat.search(stem)
        if m:
            return datetime.strptime(m.group(1), fmt)
    raise ValueError(f"Could not parse time from filename: {fname}")

def load_mask_map(mapping_txt: str) -> dict:
    """
    mapping_txt lines:
        2018-05-01-00-02-44_logits.npz,mask_2018-05-01-00-02-44.npy
    Returns dict: { '2018-..._logits.npz': 'mask_2018-....npy', ... }
    """
    mp = {}
    with open(mapping_txt, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2:
                continue
            mp[parts[0]] = parts[1]
    if not mp:
        raise ValueError(f"Empty or invalid mapping file: {mapping_txt}")
    return mp

# ---------- dataset ----------

class CloudLogitsDataset(Dataset):
    """
    Loads sequences of logits images with per-frame masks.
    Each sequence is length T (sequence_length) with a fixed sample interval in minutes.
    Returns:
      - video: FloatTensor [C, T, H, W] in [-0.5, 0.5]
      - mask_video: FloatTensor [T, H, W] in {0,1}
      - times: list[datetime] for debugging/eval
      - files: list[str] npz paths
    """
    def __init__(
        self,
        logits_dir: str,
        mask_dir: str,
        mapping_txt: str,
        slots: list,                      # [[start, end], ...], strings
        image_size: int = 64,
        sequence_length: int = 8,
        sample_interval_minutes: int = 60,
        is_training: bool = True
    ):
        super().__init__()
        self.logits_dir = logits_dir
        self.mask_dir = mask_dir
        self.mapping_txt = mapping_txt
        self.mask_map = load_mask_map(mapping_txt)
        self.image_size = int(image_size)
        self.sequence_length = int(sequence_length)
        self.interval = timedelta(minutes=int(sample_interval_minutes))
        self.is_training = is_training

        # index all npz
        pattern = os.path.join(self.logits_dir, "**", "*.npz")
        all_npz = sorted(glob(pattern, recursive=True))
        if not all_npz:
            raise ValueError(f"No .npz files found under {self.logits_dir}")

        # extract (time, path) for valid ones
        time_files = []
        for p in all_npz:
            try:
                t = parse_time_from_filename(p)
                time_files.append((t, p))
            except Exception:
                # skip files without parseable time
                continue
        time_files.sort(key=lambda x: x[0])
        if not time_files:
            raise ValueError("No timestamp-parsable files found.")

        # filter by multi-slots
        if not isinstance(slots, (list, tuple)) or not slots:
            raise ValueError("slots must be a non-empty list of [start,end] pairs.")
        self.filtered = []
        for start, end in slots:
            s = parse_time_from_string(start)
            e = parse_time_from_string(end)
            for t, p in time_files:
                if s <= t <= e:
                    self.filtered.append((t, p))
        self.filtered.sort(key=lambda x: x[0])
        if len(self.filtered) < self.sequence_length:
            raise ValueError("Not enough files inside provided time slots.")
        print(f"totally find {len(self.filtered)} slots in between")
        # build sequences with continuity check
        self.sequences = []
        i = 0
        # allow small drift of +/- 60 minutes per step
        tolerance = timedelta(minutes=60)
        while i <= len(self.filtered) - self.sequence_length:
            ok = True
            win = self.filtered[i:i+self.sequence_length]
            for j in range(1, self.sequence_length):
                if abs((win[j][0] - win[j-1][0]) - self.interval) > tolerance:
                    ok = False
                    break
            if ok:
                self.sequences.append(win)
                # non-overlapping or sliding window? Choose sliding for more data:
                i += self.sequence_length#1
            else:
                i += 1

        if not self.sequences:
            raise ValueError("No valid sequences could be formed from the slots + interval.")

        print(f"[CloudLogitsDataset] {len(self.sequences)} sequences @ {self.sequence_length} frames each.")

    # ---- helpers ----

    def _load_frame_npz(self, path_npz: str) -> np.ndarray:
        """
        Loads the first 2D/3D array found in an .npz and converts to HxWxC.
        Normalizes to [0,1] then shifts to [-0.5,0.5]
        """
        arr = None
        with np.load(path_npz) as data:
            for k in data.files:
                a = data[k]
                if a.ndim >= 2:
                    arr = a
                    break
        if arr is None:
            # fallback empty
            arr = np.zeros((self.image_size, self.image_size), np.float32)

        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)  # H,W -> H,W,3
        elif arr.ndim == 3:
            # unify to H,W,C
            if arr.shape[0] in (1,3,4):  # C,H,W -> H,W,C
                arr = np.transpose(arr, (1,2,0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]

        # resize to 64x64
        # print(arr)
        # # arr = cv2.resize(arr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        # arr = arr.astype(np.float32)
        # if arr.max() > 1.0 or arr.min() < 0.0:
        #     arr = np.clip(arr, 0, 255) / 255.0
        # arr = arr - 0.5
        # max_arr = np.max(arr)
        # min_arr = np.min(arr)
        # delta_arr = max_arr - min_arr
        # arr = (arr-min_arr)/delta_arr - 0.5
        # m = np.mean(arr)
        # s = np.std(arr)
        # arr = (arr-m)/s/6
        return arr  # H,W,3 in [-0.5,0.5]

    def _mask_for_npz(self, path_npz: str) -> np.ndarray:
        """
        Loads mask (H,W) from the mapping. If missing, returns ones.
        """
        base = os.path.basename(path_npz)
        mask_name = self.mask_map.get(base, None)
        if not mask_name:
            # no mapping -> ones
            return np.ones((self.image_size, self.image_size), dtype=np.float32)
        path_mask = os.path.join(self.mask_dir, mask_name)
        if not os.path.exists(path_mask):
            # mapping exists but file missing -> ones
            print("mask missing, using ones")
            return np.ones((self.image_size, self.image_size), dtype=np.float32)
        m = np.load(path_mask)
        if m.ndim > 2:
            m = m[..., 0]
        # ensure 64x64
        if m.shape[:2] != (self.image_size, self.image_size):
            m = cv2.resize(m.astype(np.float32), (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        # binary in {0,1}
        m = (m > 0.5).astype(np.float32)
        return m  # H,W

    # ---- PyTorch Dataset API ----

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        win = self.sequences[idx]  # list of (time, path)
        imgs = []
        masks = []
        times = []
        files = []
        for t, p in win:
            img = self._load_frame_npz(p)    # H,W,C [-0.5,0.5]
            m = self._mask_for_npz(p)        # H,W
            imgs.append(img*m.reshape(64,64,1))
            masks.append(m)
            times.append(t)
            files.append(p)

        # stack to [T,H,W,C] and [T,H,W]
        video = np.stack(imgs, axis=0)
        me = np.mean(video)
        s = np.std(video)
        video = (video - me) / s / 6
        
        mask_video = np.stack(masks, axis=0)

        # to torch: [C,T,H,W]
        video = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous().float()
        mask_video = torch.from_numpy(mask_video).contiguous().float()
        data = {
            "video": video,             # [C, T, H, W]
            "mask_video": mask_video,   # [T, H, W]
            #"times": times,
            "files": files
        }
        #print(data)
        return data

# ---------- loaders ----------

def create_data_loaders_from_json(cfg: dict, batch_size_vqvae: int, batch_size_gpt: int):
    """
    cfg["data"] contains:
        logits_dir, mask_dir, mapping_txt, image_size, sequence_length,
        sample_interval_minutes, train, test
    """
    D = cfg["data"]

    train_ds = CloudLogitsDataset(
        logits_dir=D["logits_dir"],
        mask_dir=D["mask_dir"],
        mapping_txt=D["mapping_txt"],
        slots=D["train"],
        image_size=D.get("image_size", 64),
        sequence_length=D.get("sequence_length", 8),
        sample_interval_minutes=D.get("sample_interval_minutes", 5),
        is_training=True
    )

    val_ds = CloudLogitsDataset(
        logits_dir=D["logits_dir"],
        mask_dir=D["mask_dir"],
        mapping_txt=D["mapping_txt"],
        slots=D["val"],
        image_size=D.get("image_size", 64),
        sequence_length=D.get("sequence_length", 8),
        sample_interval_minutes=D.get("sample_interval_minutes", 5),
        is_training=False
    )

    train_loader_vqvae = DataLoader(
        train_ds, batch_size=batch_size_vqvae, shuffle=True, num_workers= D.get("num_workers", 2), pin_memory=True
    )
    val_loader_vqvae = DataLoader(
        val_ds, batch_size=batch_size_vqvae, shuffle=False, num_workers= D.get("num_workers", 2), pin_memory=True
    )

    # GPT can use different batch size if desired
    train_loader_gpt = DataLoader(
        train_ds, batch_size=batch_size_gpt, shuffle=True, num_workers= D.get("num_workers", 2), pin_memory=True
    )
    val_loader_gpt = DataLoader(
        val_ds, batch_size=batch_size_gpt, shuffle=False, num_workers= D.get("num_workers", 2), pin_memory=True
    )

    return (train_loader_vqvae, val_loader_vqvae), (train_loader_gpt, val_loader_gpt)
