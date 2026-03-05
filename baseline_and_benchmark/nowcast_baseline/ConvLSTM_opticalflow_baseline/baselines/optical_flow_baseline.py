
# baselines/optical_flow_baseline.py
# Parameterizable Farneback optical flow predictor for per-channel warping.
# Provides:
#   - class OpticalFlowPredictor (with .predict_next on logits)
#   - helper warp utilities
#
from typing import Dict, Any, Tuple
import numpy as np
import cv2

def _to_gray_from_logits(frame_logits: np.ndarray, mode: str = "argmax", flow_channel: int = 1) -> np.ndarray:
    """
    frame_logits: [C, H, W]
    returns grayscale float32 [H, W] in range [0,1]
    """
    C, H, W = frame_logits.shape
    if mode == "channel":
        c = np.clip(flow_channel, 0, C-1)
        g = frame_logits[c]
        g = g - g.min()
        if g.max() > 0:
            g = g / (g.max() + 1e-6)
        return g.astype(np.float32)
    else:
        # argmax -> labels in [0..C-1], normalize
        labels = frame_logits.argmax(axis=0).astype(np.float32)
        if C > 1:
            labels = labels / (C - 1)
        return labels

def warp_with_flow(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    img: [H,W] or [C,H,W]
    flow: [H,W,2] (dx, dy)
    returns: warped image with same shape as img
    """
    H, W = flow.shape[:2]
    grid_y, grid_x = np.mgrid[0:H, 0:W].astype(np.float32)
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]
    if img.ndim == 2:
        return cv2.remap(img.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    else:
        C = img.shape[0]
        out = np.zeros_like(img, dtype=np.float32)
        for c in range(C):
            out[c] = cv2.remap(img[c].astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out

class OpticalFlowPredictor:
    """
    Simple Farneback optical flow on the last two frames, warping the last frame to the next time.
    """
    def __init__(self, params: Dict[str, Any] = None):
        # Defaults close to OpenCV docs, can be tuned
        p = params or {}
        self.pyr_scale = float(p.get("pyr_scale", 0.5))
        self.levels = int(p.get("levels", 3))
        self.winsize = int(p.get("winsize", 15))
        self.iterations = int(p.get("iterations", 3))
        self.poly_n = int(p.get("poly_n", 5))
        self.poly_sigma = float(p.get("poly_sigma", 1.2))
        self.flags = int(p.get("flags", 0))
        self.gray_mode = p.get("gray_mode", "channel")  # "channel" or "argmax"
        self.flow_channel = int(p.get("flow_channel", 1))  # used when gray_mode == "channel"

    def compute_flow(self, prev_logits: np.ndarray, curr_logits: np.ndarray) -> np.ndarray:
        """
        prev_logits, curr_logits: [C,H,W]
        returns flow [H,W,2] in float32
        """
        prev_gray = _to_gray_from_logits(prev_logits, mode=self.gray_mode, flow_channel=self.flow_channel)
        curr_gray = _to_gray_from_logits(curr_logits, mode=self.gray_mode, flow_channel=self.flow_channel)
        flow = cv2.calcOpticalFlowFarneback(
            prev=prev_gray, next=curr_gray, flow=None,
            pyr_scale=self.pyr_scale, levels=self.levels, winsize=self.winsize,
            iterations=self.iterations, poly_n=self.poly_n, poly_sigma=self.poly_sigma, flags=self.flags
        )
        return flow

    def predict_next(self, input_logits: np.ndarray) -> np.ndarray:
        """
        input_logits: [T,C,H,W] (T>=2). Uses last two frames to compute flow and warps the last frame.
        returns pred_logits: [C,H,W]
        """
        assert input_logits.ndim == 4 and input_logits.shape[0] >= 2, "need at least two frames"
        prev = input_logits[-2]  # [C,H,W]
        curr = input_logits[-1]
        flow = self.compute_flow(prev, curr)    # [H,W,2]
        pred = warp_with_flow(curr, flow)       # [C,H,W]
        return pred.astype(np.float32)
