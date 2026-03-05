"""
Code to inference the next frames using a trained VideoGPT model.
"""
import json, torch
import numpy as np
from typing import Union, Optional
from vqvae import VQVAE
from transformer import VideoGPT

class CloudNowcaster:
    def __init__(self, config_json: str, vqvae_ckpt: str, gpt_ckpt: str, device: Optional[str] = None):
        """
        config_json has same structure as training JSON (we need vocab size for GPT).
        """
        with open(config_json, "r") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # VQ-VAE
        vqcfg = {"model": cfg["vqvae"]["model"], "training": cfg["vqvae"]["training"]}
        self.vqvae = VQVAE(vqcfg).to(self.device)
        if vqvae_ckpt.endswith(".ckpt"):
            sd = torch.load(vqvae_ckpt, map_location="cpu")["state_dict"]
        else:
            sd = torch.load(vqvae_ckpt, map_location="cpu")
        self.vqvae.load_state_dict(sd)
        self.vqvae.eval()

        # GPT
        self.gpt = VideoGPT(cfg["gpt"]).to(self.device)

        # predictor.py — replace the GPT-loading block
        try:
            gsd_raw = torch.load(gpt_ckpt, map_location="cpu", weights_only=True)
        except TypeError:
            # older torch without weights_only
            gsd_raw = torch.load(gpt_ckpt, map_location="cpu")

        sd = gsd_raw.get("state_dict", gsd_raw)

        # keep only GPT weights and strip "model." prefix from Lightning wrapper
        gpt_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                gpt_sd[k[len("model."):]] = v  # e.g. model.pos_emb -> pos_emb
            elif k.startswith(("tok_emb", "pos_emb", "transformer", "ln_f", "head")):
                # support raw VideoGPT state_dict too
                gpt_sd[k] = v

        self.gpt.load_state_dict(gpt_sd, strict=True)
        self.gpt.eval()
        # gsd = torch.load(gpt_ckpt, map_location="cpu")
        # self.gpt.load_state_dict(gsd["state_dict"] if "state_dict" in gsd else gsd)
        # self.gpt.eval()

    @torch.no_grad()
    def predict_next(
        self,
        past_frames: Union[np.ndarray, torch.Tensor],  # [T_past,H,W] or [T_past,H,W,C] or [C,T_past,H,W]
        m_future: int,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.95
    ) -> np.ndarray:
        """
        Returns predicted future m frames as numpy array [m, H, W, C] in [-0.5,0.5].
        """
        x = past_frames
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.float()
        # unify to [1,C,T,H,W] in [-0.5,0.5]
        if x.dim() == 3:            # [T,H,W]
            x = x.unsqueeze(-1).repeat(1,1,1,3)  # -> [T,H,W,3]
            x = x.permute(3,0,1,2)               # [C,T,H,W]
        elif x.dim() == 4 and x.size(-1) in (1,3):  # [T,H,W,C]
            x = x.permute(3,0,1,2)               # [C,T,H,W]
        elif x.dim() == 4 and x.size(0) in (1,3):   # [C,T,H,W]
            pass
        else:
            raise ValueError("Unsupported input shape for past_frames.")
        x = x.unsqueeze(0).to(self.device)

        # encode past -> tokens
        idx = self.vqvae.encode(x)      # [1, T_lat, h, w]
        T_lat, h, w = self.vqvae.latent_shape
        Lp = T_lat * h * w
        past_ids = idx.view(1, -1)      # [1, Lp]

        # how many tokens correspond to m future frames?
        L_future = m_future * h * w

        # generate
        full_ids = self.gpt.generate(
            input_ids=past_ids,
            max_new_tokens=L_future,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )  # [1, Lp + L_future]

        # take last m frames of tokens and decode
        tail = full_ids[:, -L_future:]             # [1, m*h*w]
        tail_idx = tail.view(1, m_future, h, w)    # [1, m, h, w]
        rec = self.vqvae.decode(tail_idx)          # [1, C, m, H, W]
        rec = rec.squeeze(0).permute(1,2,3,0).contiguous().cpu().numpy()  # [m,H,W,C]
        return rec
