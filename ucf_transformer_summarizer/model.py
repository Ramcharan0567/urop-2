import torch
from torch import nn

class TemporalTransformerSummarizer(nn.Module):
    def __init__(self, input_dim=4096, d_model=256, nhead=8, layers=3,
                 dropout=0.2, max_segments=128, top_fraction=0.25):
        super().__init__()
        self.top_fraction = top_fraction
        self.norm = nn.LayerNorm(input_dim)
        self.project = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_segments, d_model))
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout,
                                         activation="gelu", batch_first=True,
                                         norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model//2),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model//2, 1))

    def forward(self, x, lengths=None):
        b, t, _ = x.shape
        if t > self.pos.shape[1]:
            raise ValueError(f"Sequence length {t} exceeds max_segments={self.pos.shape[1]}")
        if lengths is None:
            lengths = torch.full((b,), t, device=x.device, dtype=torch.long)
        # Mask padded temporal positions so they cannot influence real segments.
        padding_mask = torch.arange(t, device=x.device)[None, :] >= lengths[:, None]
        h = self.project(self.norm(x)) + self.pos[:, :t]
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        seg = self.head(h).squeeze(-1)
        video = []
        for i in range(b):
            n = max(1, min(int(lengths[i].item()), t))
            k = max(1, int(n*self.top_fraction + 0.999999))
            video.append(torch.topk(seg[i, :n], min(k, n)).values.mean())
        return torch.stack(video), seg
