
# infer_convlstm_benchmark.py
"""
Benchmark a trained ConvLSTM with benchmark_prediction.benchmark_prediction.
This script wraps the model with a function f(input_logits, input_mask) that matches the benchmark API.
"""
import os, json, argparse
import numpy as np
import torch
from baselines.convlstm_baseline import load_convlstm_from_checkpoint
from benchmark_prediction import benchmark_prediction

def build_wrapper(ckpt_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_convlstm_from_checkpoint(ckpt_path, map_location="cpu")
    model.to(device).eval()
    n_input = int(meta.get("n_input", 2))

    @torch.no_grad()
    def f(input_logits: np.ndarray, input_mask: np.ndarray) -> np.ndarray:
        """
        input_logits: [T,C,H,W] raw logits
        input_mask:   [T,H,W]   binary mask
        returns:      [C,H,W]   predicted logits (same scale as inputs)
        """
        # mask + normalise like training (dataset scales by z = (x - mean)/std/6)
        T, C, H, W = input_logits.shape
        assert T == n_input, f"Expected T={n_input}, got {T}."
        mask4 = input_mask.reshape(T, 1, H, W).astype(np.float32)
        data = input_logits.astype(np.float32) * mask4
        # m = float(data.mean())
        # s = float(data.std() + 1e-6)
        # data_n = (data - m) / s / 6.0                       # [T,C,H,W]
        x = torch.from_numpy(data).permute(1,0,2,3).unsqueeze(0).to(device)  # [1,C,T,H,W]
        y_hat = model(x).squeeze(0).cpu().numpy()           # [C,H,W] in normalised space
        # y_hat = y_hat * s * 6.0 + m
        return y_hat.astype(np.float32)
    return f, n_input

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="config JSON for benchmarking (period, dirs, class_map, etc.)")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = json.load(f)

    ckpt = cfg["checkpoint"]
    f, n_input = build_wrapper(ckpt)
    metrics = benchmark_prediction(f, config=cfg, n_step=n_input)
    print(json.dumps(metrics, indent=2))

    out_dir = cfg.get("out_dir", "./benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg.get("metrics_name", "ConvLSTM_metrics.json"))
    with open(out_path, "w") as g:
        json.dump(metrics, g, indent=2)
    print(f"Saved metrics to {out_path}")

if __name__ == "__main__":
    main()
