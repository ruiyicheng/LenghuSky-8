
# train_convlstm.py
"""
Train a ConvLSTM next-frame predictor on CloudLogitsDataset sequences.
The dataset & normalisation scheme follow loader.CloudLogitsDataset.

Config JSON (example in config_convlstm_train.json) includes:
{
  "seed": 42,
  "data": {... like config_train.json ...},
  "convlstm": {
     "model": {"hidden_dims": [64,64], "kernel_size": 3, "dropout": 0.0},
     "training": {"batch_size": 8, "learning_rate": 1e-3, "max_epochs": 5, "save_dir": "./checkpoints/convlstm"},
     "n_input": 2
  }
}
"""
import os, json, math, random, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from baselines.convlstm_baseline import ConvLSTMNet, masked_mse
from loader import CloudLogitsDataset

def set_seed(seed: int):
    import random
    import numpy as np
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_dataloaders(cfg: dict, n_input: int):
    D = cfg["data"]
    seq_len = int(D.get("sequence_length", n_input + 1))
    if seq_len != n_input + 1:
        # enforce consistent length for next-frame prediction
        seq_len = n_input + 1
    train_ds = CloudLogitsDataset(
        logits_dir=D["logits_dir"],
        mask_dir=D["mask_dir"],
        mapping_txt=D["mapping_txt"],
        slots=D["train"],
        image_size=D.get("image_size", 64),
        sequence_length=seq_len,
        sample_interval_minutes=D.get("sample_interval_minutes", 60),
        is_training=True
    )
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
    bs = cfg["convlstm"]["training"]["batch_size"]
    nw = D.get("num_workers", 2)
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=nw, pin_memory=True),
        DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    )

def train_one_epoch(model, loader, device, optim):
    model.train()
    total, n = 0.0, 0
    total_sample = len(loader)
    for batch in loader:
        
        video = batch["video"].to(device)           # [B,C,T,H,W] (already normalised in dataset)
        maskv = batch["mask_video"].to(device)      # [B,T,H,W]
        # create (input, target, mask) for next-frame prediction
        x = video[:, :, :-1, :, :]                  # [B,C,n_input,H,W]
        y = video[:, :, -1,  :, :]                  # [B,C,H,W]
        m = maskv[:, -1, :, :]                      # [B,H,W] (target mask)
        y_hat = model(x)                            # [B,C,H,W]
        loss = masked_mse(y_hat, y, m)
        optim.zero_grad(); loss.backward(); optim.step()
        total += float(loss.detach().cpu().item()); n += 1
    return total / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        video = batch["video"].to(device)
        maskv = batch["mask_video"].to(device)
        x = video[:, :, :-1, :, :]
        y = video[:, :, -1,  :, :]
        m = maskv[:, -1, :, :]
        y_hat = model(x)
        loss = masked_mse(y_hat, y, m)
        total += float(loss.detach().cpu().item()); n += 1
    return total / max(n,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_input = int(cfg["convlstm"].get("n_input", 2))
    loaders = make_dataloaders(cfg, n_input)
    train_loader, val_loader = loaders

    in_channels = 3  # your logits have C=3
    md = cfg["convlstm"]["model"]
    model = ConvLSTMNet(
        in_channels=in_channels,
        hidden_dims=md.get("hidden_dims", [64,64]),
        kernel_size=md.get("kernel_size", 3),
        dropout=md.get("dropout", 0.0)
    ).to(device)

    tr = cfg["convlstm"]["training"]
    lr = float(tr.get("learning_rate", 1e-3))
    max_epochs = int(tr.get("max_epochs", 5))
    save_dir = tr.get("save_dir", "./checkpoints/convlstm")
    os.makedirs(save_dir, exist_ok=True)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val = float("inf"); best_path = None
    for e in range(1, max_epochs+1):
        tr_loss = train_one_epoch(model, train_loader, device, optim)
        va_loss = evaluate(model, val_loader, device)
        print(f"[Epoch {e:03d}] train_loss={tr_loss:.6f}  val_loss={va_loss:.6f}")
        if va_loss < best_val:
            best_val = va_loss
            best_path = os.path.join(save_dir, f"convlstm-epoch={e:02d}-val={va_loss:.6f}.pt")
            meta = {
                "in_channels": in_channels,
                "hidden_dims": md.get("hidden_dims", [64,64]),
                "kernel_size": md.get("kernel_size", 3),
                "dropout": md.get("dropout", 0.0),
                "n_input": n_input
            }
            torch.save({"state_dict": model.state_dict(), "meta": meta}, best_path)
            print(f"  -> saved {best_path}")
    print(f"Best val={best_val:.6f} at {best_path}")

if __name__ == "__main__":
    main()
