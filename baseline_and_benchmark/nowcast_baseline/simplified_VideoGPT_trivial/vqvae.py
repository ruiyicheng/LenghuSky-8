# vqvae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

# ---- simple residual blocks and 2D time-sliced encoder/decoder (kept from your design) ----

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + res
        return self.relu(x)

class VideoEncoder(nn.Module):
    """Encode [B,C,T,H,W] by slicing time to [B*T,C,H,W] -> [B,D,T,H',W']"""
    def __init__(self, in_channels=3, hidden_dims=[64,128,256], out_channels=256):
        super().__init__()
        layers = []
        c = in_channels
        for h in hidden_dims:
            layers += [
                nn.Conv2d(c, h, 4, stride=2, padding=1),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True),
                ResidualBlock(h, h)
            ]
            c = h
        layers += [nn.Conv2d(c, out_channels, 3, padding=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        B, C, T, H, W = x.shape
        xt = x.permute(0,2,1,3,4).contiguous().view(B*T, C, H, W)     # [B*T,C,H,W]
        z = self.encoder(xt)                                          # [B*T,D,h,w]
        _, D, h, w = z.shape
        z = z.view(B, T, D, h, w).permute(0,2,1,3,4).contiguous()     # [B,D,T,h,w]
        return z

class VideoDecoder(nn.Module):
    """Decode [B,D,T,h,w] -> [B,C,T,H,W] by slicing time to [B*T,D,h,w]"""
    def __init__(self, out_channels=3, hidden_dims=[64,128,256], in_channels=256):
        super().__init__()
        layers = []
        c = in_channels
        for h in reversed(hidden_dims):
            layers += [
                nn.ConvTranspose2d(c, h, 4, stride=2, padding=1),
                nn.BatchNorm2d(h),
                nn.ReLU(inplace=True),
                ResidualBlock(h, h)
            ]
            c = h
        layers += [nn.Conv2d(c, out_channels, 3, padding=1), nn.Tanh()]
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        B, D, T, h, w = z.shape
        zt = z.permute(0,2,1,3,4).contiguous().view(B*T, D, h, w)     # [B*T,D,h,w]
        x = self.decoder(zt)                                          # [B*T,C,H,W]
        _, C, H, W = x.shape
        x = x.view(B, T, C, H, W).permute(0,2,1,3,4).contiguous()     # [B,C,T,H,W]
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):  # z: [B,D,T,h,w]
        B, D, T, h, w = z.shape
        z_flat = z.permute(0,2,3,4,1).contiguous().view(B*T*h*w, D)   # [N,D]
        # distances
        e = self.embedding.weight                                     # [K,D]
        d = (z_flat**2).sum(1, keepdim=True) + (e**2).sum(1) - 2*z_flat @ e.t()
        idx = d.argmin(dim=1)                                         # [N]
        z_q = self.embedding(idx).view(B, T, h, w, D).permute(0,4,1,2,3).contiguous()  # [B,D,T,h,w]
        # losses
        commit = F.mse_loss(z_q.detach(), z)
        codebk = F.mse_loss(z_q, z.detach())
        vq_loss = codebk + self.commitment_cost * commit
        # straight-through
        z_q = z + (z_q - z).detach()
        # perplexity
        onehot = F.one_hot(idx, self.num_embeddings).float()
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * (avg_probs+1e-10).log()).sum())
        return z_q, vq_loss, idx, perplexity

# ---- VQVAE ----

class VQVAE(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = config
        md = config["model"]
        tr = config["training"]

        self.embedding_dim = md["embedding_dim"]
        self.num_embeddings = md["n_codes"]
        self.commitment_cost = md.get("commitment_cost", 0.25)

        self.encoder = VideoEncoder(
            in_channels=3,
            hidden_dims=md.get("hidden_dims", [64,128,256]),
            out_channels=self.embedding_dim
        )
        self.vq = VectorQuantizer(self.num_embeddings, self.embedding_dim, self.commitment_cost)
        self.decoder = VideoDecoder(
            out_channels=3,
            hidden_dims=md.get("hidden_dims", [64,128,256]),
            in_channels=self.embedding_dim
        )

        # filled on first encode
        self.latent_shape = None  # (T_lat, H_lat, W_lat)

        self.lr = tr["learning_rate"]
        self.max_epochs = tr.get("max_epochs", 50)

    def forward(self, x):  # x: [B,C,T,H,W]
        z = self.encoder(x)                            # [B,D,T,h,w]
        z_q, vq_loss, idx, perplexity = self.vq(z)    # idx is flat [B*T*h*w]
        x_rec = self.decoder(z_q)                      # [B,C,T,H,W]
        # cache latent shape
        _, _, T, h, w = z.shape
        self.latent_shape = (T, h, w)
        return x_rec, vq_loss, idx, perplexity

    def training_step(self, batch, batch_idx):
        x = batch["video"]          # [B,C,T,H,W]
        mask_video = batch["mask_video"]  # [B,T,H,W]
        x_rec, vq_loss, _, ppl = self(x)
        recon = self._masked_mse(x_rec, x, mask_video)
        loss = recon + vq_loss
        self.log_dict({"train_loss": loss, "train_rec": recon, "train_vq": vq_loss, "train_ppl": ppl}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        mask_video = batch["mask_video"]
        x_rec, vq_loss, _, ppl = self(x)
        recon = self._masked_mse(x_rec, x, mask_video)
        loss = recon + vq_loss
        self.log_dict({"val_loss": loss, "val_rec": recon, "val_vq": vq_loss, "val_ppl": ppl}, prog_bar=True)
        return loss

    @staticmethod
    def _masked_mse(pred, target, mask_video):
        """
        pred, target: [B,C,T,H,W]
        mask_video:   [B,T,H,W]  (binary 0/1)
        """
        B, C, T, H, W = pred.shape
        if mask_video.dim() == 3:   # [T,H,W] -> [B,T,H,W]
            mask_video = mask_video.unsqueeze(0).repeat(B,1,1,1)
        # resize mask to H,W if mismatch
        m = mask_video
        if m.shape[-2:] != (H,W):
            m = F.interpolate(m.unsqueeze(1), size=(H,W), mode='nearest').squeeze(1)  # [B,T,H,W]
        m = m.unsqueeze(1).repeat(1, C, 1, 1, 1)  # [B,C,T,H,W]
        return F.mse_loss(pred * m, target * m)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        return [opt], [sch]

    # ---- public encode/decode ----

    def encode(self, x):  # x: [B,C,T,H,W] or [B,C,H,W] (single frame)
        if x.dim() == 4:
            x = x.unsqueeze(2)
        z = self.encoder(x)                 # [B,D,T,h,w]
        _, _, T, h, w = z.shape
        self.latent_shape = (T, h, w)
        _, _, idx, _ = self.vq(z)           # idx: [B*T*h*w] flat
        idx = idx.view(x.size(0), T, h, w)  # [B,T,h,w]
        return idx

    def decode(self, idx):  # idx: [B,T,h,w] of ints in [0..K-1]
        B, T, h, w = idx.shape
        onehot = F.one_hot(idx.view(-1), self.num_embeddings).float()  # [B*T*h*w, K]
        z_flat = onehot @ self.vq.embedding.weight                     # [N, D]
        z = z_flat.view(B, T, h, w, self.embedding_dim).permute(0,4,1,2,3).contiguous()  # [B,D,T,h,w]
        return self.decoder(z)  # [B,C,T,H,W]
