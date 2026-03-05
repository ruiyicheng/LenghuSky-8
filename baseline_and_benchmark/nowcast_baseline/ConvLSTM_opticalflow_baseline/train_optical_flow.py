
# train_optical_flow.py
"""
"Training" for optical flow is a light hyperparameter search on a validation split.
We pick Farneback parameters that minimise masked MSE between warped last-input and target.

Config JSON (example in config_optflow_train.json):
{
  "seed": 42,
  "data": {... like config_train.json ...},
  "optflow": {
    "n_input": 2,
    "search_space": {
      "winsize": [9,13,15],
      "levels": [3,4],
      "iterations": [3,5],
      "poly_n": [5,7],
      "poly_sigma": [1.1,1.3],
      "pyr_scale": [0.5]
    },
    "gray_mode": "channel",
    "flow_channel": 1,
    "training": {"max_eval_samples": 512, "save_dir": "./checkpoints/optflow"}
  }
}
"""
import os, json, argparse, itertools, numpy as np, torch
from torch.utils.data import DataLoader
from loader import CloudLogitsDataset
from baselines.optical_flow_baseline import OpticalFlowPredictor
from baselines.convlstm_baseline import masked_mse

def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate_params(ds, params, n_input: int, max_samples: int = 256) -> float:
    pr = OpticalFlowPredictor(params)
    total, n = 0.0, 0
    for i in range(min(len(ds), max_samples)):
        item = ds[i]
        video = item["video"].numpy()       # [C,T,H,W] normalised (but we only warp raw last frame semantics)
        masks = item["mask_video"].numpy()  # [T,H,W]
        # denormalisation constants computed like dataset
        # dataset did: video_norm = (video_raw * mask - m)/s/6
        # We do not have raw values here (only normalized), so we compare in normalized space to be consistent.
        x = video[:, :-1, :, :]             # [C,n_input,H,W]
        y = video[:, -1,  :, :]             # [C,H,W]
        m = masks[-1, :, :]
        # predictor expects [T,C,H,W] in *original* scale; but we don't have it now, we can still operate on normalized
        xin = np.transpose(x, (1,0,2,3))    # [T,C,H,W]
        y_hat = pr.predict_next(xin)        # [C,H,W]
        # masked MSE in the dataset's scale (normalized)
        loss = float(((y_hat - y) ** 2 * m).mean())
        total += loss; n += 1
    return total / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)
    set_seed(int(cfg.get("seed", 42)))

    D = cfg["data"]
    OF = cfg["optflow"]
    n_input = int(OF.get("n_input", 2))
    seq_len = n_input + 1

    # we only need a validation set to tune on
    val_ds = CloudLogitsDataset(
        logits_dir=D["logits_dir"],
        mask_dir=D["mask_dir"],
        mapping_txt=D["mapping_txt"],
        slots=D["val"],
        image_size=D.get("image_size", 64),
        sequence_length=seq_len,
        sample_interval_minutes=D.get("sample_interval_minutes", 60),
        is_training=False
    )

    space = OF.get("search_space", {})
    keys = ["winsize", "levels", "iterations", "poly_n", "poly_sigma", "pyr_scale"]
    values = [space.get(k, [v]) for k, v in zip(keys, [15,3,3,5,1.2,0.5])]
    grid = list(itertools.product(*values))

    best_score, best_params = float("inf"), None
    max_eval = int(OF.get("training", {}).get("max_eval_samples", 256))
    for tup in grid:
        params = dict(zip(keys, tup))
        params["gray_mode"] = OF.get("gray_mode", "channel")
        params["flow_channel"] = OF.get("flow_channel", 1)
        score = evaluate_params(val_ds, params, n_input=n_input, max_samples=max_eval)
        print(f"Params={params} -> val_mse={score:.6f}")
        if score < best_score:
            best_score, best_params = score, params

    save_dir = OF.get("training", {}).get("save_dir", "./checkpoints/optflow")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"optflow-best.json")
    meta = {"n_input": n_input, "best_params": best_params, "val_mse": best_score}
    with open(path, "w") as g:
        json.dump(meta, g, indent=2)
    print(f"Saved best params to {path}")

if __name__ == "__main__":
    main()
