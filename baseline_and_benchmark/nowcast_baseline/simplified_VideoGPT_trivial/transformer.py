# transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

def causal_mask(sz: int, device=None):
    # lower-triangular causal mask for attention
    mask = torch.full((sz, sz), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask

class VideoGPT(nn.Module):
    """
    GPT-style LM over VQ-VAE code indices.
    Expect input_ids: [B, L] where L = T_lat * H_lat * W_lat (or prefix length during generation).
    """
    def __init__(self, config):
        super().__init__()
        md = config["model"]
        self.vocab_size = md["n_codes"]
        self.d_model = md["d_model"]
        self.nhead = md["nhead"]
        self.num_layers = md["num_layers"]
        self.dropout = md.get("dropout", 0.1)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, 20000, self.d_model))  # enough positions
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead, dropout=self.dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=self.num_layers)
        self.ln_f = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, input_ids):  # [B,L]
        B, L = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb[:, :L, :]
        mask = causal_mask(L, device=x.device)
        x = self.transformer(x, mask=mask)
        x = self.ln_f(x)
        logits = self.head(x)  # [B,L,V]
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        Autoregressive sampling. input_ids: [B, L0]
        Returns [B, L0 + max_new_tokens]
        """
        B = input_ids.size(0)
        seq = input_ids
        for _ in range(max_new_tokens):
            L = seq.size(1)
            x = self.tok_emb(seq) + self.pos_emb[:, :L, :]
            mask = causal_mask(L, device=seq.device)
            x = self.transformer(x, mask=mask)
            x = self.ln_f(x)
            logits = self.head(x)[:, -1, :] / max(temperature, 1e-5)  # [B,V]

            if top_k is not None:
                topk = torch.topk(logits, top_k, dim=-1).values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < topk, torch.full_like(logits, float("-inf")), logits)

            if top_p is not None and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                cutoff = (cumprobs > top_p).float().argmax(dim=-1)
                # mask tokens beyond nucleus
                for b in range(B):
                    k = cutoff[b].item()
                    sorted_logits[b, k+1:] = float("-inf")
                # unsort
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            #next_id = torch.multinomial(probs, num_samples=1)  # [B,1]
            next_id = torch.argmax(probs, dim=-1, keepdim=True)  # [B,1]
            seq = torch.cat([seq, next_id], dim=1)
        return seq

# ---- Lightning wrapper ----

class VideoGPTLightning(pl.LightningModule):
    def __init__(self, config, vqvae):
        super().__init__()
        self.save_hyperparameters(ignore=['vqvae'])
        self.cfg = config
        self.vqvae = vqvae
        self.model = VideoGPT(config)
        self.lr = config["training"]["learning_rate"]

    def training_step(self, batch, batch_idx):
        x = batch["video"]                 # [B,C,T,H,W]
        with torch.no_grad():
            idx = self.vqvae.encode(x)     # [B,T,h,w]
        ids = idx.view(idx.size(0), -1)    # [B, L]
        inp = ids[:, :-1]
        tgt = ids[:, 1:]
        logits = self.model(inp)           # [B,L-1,V]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        with torch.no_grad():
            idx = self.vqvae.encode(x)
        ids = idx.view(idx.size(0), -1)
        inp = ids[:, :-1]
        tgt = ids[:, 1:]
        logits = self.model(inp)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return opt
