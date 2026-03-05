"""
Train both the VAE part and the Transformer part of VideoGPT model.
"""

import os, json, torch, pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from loader import create_data_loaders_from_json
from vqvae import VQVAE
from transformer import VideoGPTLightning

def main(config_path: str):
    with open(config_path, "r") as f:
        cfg = json.load(f)

    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)

    # ----- data -----
    (train_vqvae, val_vqvae), (train_gpt, val_gpt) = create_data_loaders_from_json(
        cfg,
        batch_size_vqvae=cfg["vqvae"]["training"]["batch_size"],
        batch_size_gpt=cfg["gpt"]["training"]["batch_size"]
    )

    # ----- VQ-VAE -----
    vqcfg = {"model": cfg["vqvae"]["model"], "training": cfg["vqvae"]["training"]}
    ckpt_dir_vq = cfg["vqvae"]["training"]["save_dir"]
    os.makedirs(ckpt_dir_vq, exist_ok=True)

    vq = VQVAE(vqcfg)
    cb_vq = ModelCheckpoint(dirpath=ckpt_dir_vq, filename="vqvae-{epoch:02d}-{val_loss:.4f}",
                            save_top_k=1, monitor="val_loss", mode="min")
    #es_vq = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    lrmon = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=vqcfg["training"]["max_epochs"],
        callbacks=[cb_vq, lrmon],#, es_vq
        accelerator="auto", devices=1, log_every_n_steps=1000, val_check_interval=0.99
    )
    print("=== Training VQ-VAE ===")
    trainer.fit(vq, train_vqvae, val_vqvae)
    vq_best = cb_vq.best_model_path
    print(f"VQ-VAE best: {vq_best}")

    # load best for GPT stage
    vq_loaded = VQVAE(vqcfg)
    vq_loaded.load_state_dict(torch.load(vq_best, map_location="cpu")["state_dict"] if vq_best.endswith(".ckpt")
                              else torch.load(vq_best, map_location="cpu"))
    vq_loaded.eval()

    # ----- GPT -----
    gptcfg = {"model": cfg["gpt"]["model"], "training": cfg["gpt"]["training"]}
    ckpt_dir_gpt = cfg["gpt"]["training"]["save_dir"]
    os.makedirs(ckpt_dir_gpt, exist_ok=True)

    gpt = VideoGPTLightning(cfg["gpt"], vq_loaded)
    cb_gpt = ModelCheckpoint(dirpath=ckpt_dir_gpt, filename="gpt-{epoch:02d}-{val_loss:.4f}",
                             save_top_k=1, monitor="val_loss", mode="min")
    es_gpt = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer2 = pl.Trainer(
        max_epochs=gptcfg["training"]["max_epochs"],
        callbacks=[cb_gpt, lrmon],#, es_gpt
        accelerator="auto", devices=1, log_every_n_steps=1000, val_check_interval=0.99
    )
    print("=== Training GPT ===")
    trainer2.fit(gpt, train_gpt, val_gpt)
    print(f"GPT best: {cb_gpt.best_model_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="path to JSON config")
    args = ap.parse_args()
    main(args.config)
