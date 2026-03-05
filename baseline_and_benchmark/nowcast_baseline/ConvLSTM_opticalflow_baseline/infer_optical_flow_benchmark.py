
# infer_optical_flow_benchmark.py
"""
Benchmark an optical-flow-based baseline with benchmark_prediction.benchmark_prediction.
"""
import os, json, argparse, numpy as np
from baselines.optical_flow_baseline import OpticalFlowPredictor
from benchmark_prediction import benchmark_prediction

def build_wrapper(params_or_ckpt: str, n_input: int = 2):
    # if params_or_ckpt is a JSON file, load it
    params = None
    if params_or_ckpt.endswith(".json") and os.path.exists(params_or_ckpt):
        with open(params_or_ckpt, "r") as f:
            ck = json.load(f)
        n_input = int(ck.get("n_input", n_input))
        params = ck.get("best_params", None) or ck
    else:
        # assume it's a JSON string of params or leave default
        try:
            params = json.loads(params_or_ckpt)
        except Exception:
            params = None
    predictor = OpticalFlowPredictor(params)

    def f(input_logits: np.ndarray, input_mask: np.ndarray) -> np.ndarray:
        # input_logits: [T,C,H,W]; input_mask: [T,H,W]
        # Mask last frame before warping to reduce artifacts in invalid areas.
        masked = input_logits.astype(np.float32)
        masked *= input_mask.reshape(input_mask.shape[0], 1, input_mask.shape[1], input_mask.shape[2]).astype(np.float32)
        pred = predictor.predict_next(masked)  # [C,H,W]
        return pred.astype(np.float32)
    return f, n_input

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="benchmark JSON (period, dirs, class_map, etc.)")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    src = cfg.get("checkpoint", cfg.get("params", ""))
    n_input = int(cfg.get("n_input", 2))
    f, n_input = build_wrapper(src, n_input=n_input)

    metrics = benchmark_prediction(f, config=cfg, n_step=n_input)
    print(json.dumps(metrics, indent=2))

    out_dir = cfg.get("out_dir", "./benchmark_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg.get("metrics_name", "OpticalFlow_metrics.json"))
    with open(out_path, "w") as g:
        json.dump(metrics, g, indent=2)
    print(f"Saved metrics to {out_path}")

if __name__ == "__main__":
    main()
