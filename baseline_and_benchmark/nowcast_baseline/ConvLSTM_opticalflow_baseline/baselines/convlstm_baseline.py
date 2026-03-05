
# baselines/convlstm_baseline.py
# A clean ConvLSTM implementation + utilities to train/infer on cloud logits.
# Works with loader.CloudLogitsDataset and benchmark_prediction.benchmark_prediction.
#
# Usage:
#   from baselines.convlstm_baseline import ConvLSTMNet, masked_mse
#   model = ConvLSTMNet(in_channels=3, hidden_dims=[64, 64], kernel_size=3)
#   y_hat = model(x)  # x: [B, C, T, H, W] -> y_hat: [B, C, H, W] (predict next frame)
#
# Checkpoint format saved by training script:
#   torch.save({"state_dict": ..., "meta": {...}}, path)
#
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding, bias=bias)

    def forward(self, x_t: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]):
        h_cur, c_cur = state  # [B, Hc, H, W]
        combined = torch.cat([x_t, h_cur], dim=1)
        gates = self.conv(combined)
        # split
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch: int, spatial: Tuple[int, int], device=None):
        H, W = spatial
        h = torch.zeros(batch, self.hidden_dim, H, W, device=device)
        c = torch.zeros(batch, self.hidden_dim, H, W, device=device)
        return h, c

class ConvLSTMNet(nn.Module):
    """
    Multi-layer ConvLSTM that takes a sequence [B,C,T,H,W] and predicts the next frame [B,C,H,W].
    """
    def __init__(
        self, 
        in_channels: int = 3, 
        hidden_dims: List[int] = [64, 64], 
        kernel_size: int = 3,
        num_layers: Optional[int] = None,
        dropout: float = 0.0
    ):
        super().__init__()
        if num_layers is None:
            num_layers = len(hidden_dims)
        assert num_layers == len(hidden_dims), "num_layers must match len(hidden_dims)"
        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cur_in = in_channels if i == 0 else hidden_dims[i-1]
            self.cells.append(ConvLSTMCell(cur_in, hidden_dims[i], kernel_size))
        self.proj = nn.Conv2d(hidden_dims[-1], in_channels, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W]
        returns: next frame prediction [B, C, H, W]
        """
        B, C, T, H, W = x.shape
        device = x.device
        # iterate layers
        layer_input = x  # [B,C,T,H,W] and for deeper layers it's hidden states over time
        for li in range(self.num_layers):
            cell = self.cells[li]
            # init states
            h, c = cell.init_state(B, (H, W), device=device)
            outputs_t = []
            in_ch = layer_input.size(1)
            for t in range(T):
                x_t = layer_input[:, :, t, :, :]  # [B, Cin, H, W]
                h, c = cell(x_t, (h, c))
                outputs_t.append(h)
            # stack over time for next layer input
            layer_input = torch.stack(outputs_t, dim=2)  # [B, Hc, T, H, W]
            layer_input = self.dropout(layer_input)
        # final prediction: use last hidden state of last layer
        last_h = layer_input[:, :, -1, :, :]  # [B, Hc_last, H, W]
        y = self.proj(last_h)
        return y

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: [B,C,H,W]
    mask: [B,H,W] (binary {0,1}) or broadcastable
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # [B,1,H,W]
    if mask.size(1) == 1 and pred.size(1) > 1:
        mask = mask.repeat(1, pred.size(1), 1, 1)
    return F.mse_loss(pred * mask, target * mask)

def load_convlstm_from_checkpoint(ckpt_path: str, map_location: str = "cpu") -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Returns (model, meta). The checkpoint is expected to be
      {"state_dict": state_dict, "meta": {in_channels, hidden_dims, kernel_size, dropout, n_input}}.
    """
    obj = torch.load(ckpt_path, map_location=map_location)
    state_dict = obj["state_dict"] if "state_dict" in obj else obj
    meta = obj.get("meta", {})
    model = ConvLSTMNet(
        in_channels=meta.get("in_channels", 3),
        hidden_dims=meta.get("hidden_dims", [64, 64]),
        kernel_size=meta.get("kernel_size", 3),
        dropout=meta.get("dropout", 0.0),
        num_layers=len(meta.get("hidden_dims", [64, 64]))
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta
