import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        """
        x: [B, L, C]
        returns: [B, N, d_model]
        """
        B, L, C = x.shape

        num_patches = L // self.patch_len
        x = x[:, :num_patches * self.patch_len, :]  # trim extra steps
        x = x.reshape(B, num_patches, self.patch_len * C)

        return self.proj(x)


class PatchTSTSimple(nn.Module):
    def __init__(
        self,
        input_dim,
        input_len=336,
        pred_len=96,
        patch_len=16,
        d_model=128,
        n_heads=4,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(input_dim, patch_len, d_model)

        num_patches = input_len // patch_len
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x, return_embedding=False):
        """
        x: [B, L, C]
        """
        x = self.patch_embed(x)                  # [B, N, d_model]
        x = x + self.pos_embedding[:, :x.size(1), :]
        h = self.encoder(x)                      # [B, N, d_model]

        pooled = h.mean(dim=1)                   # [B, d_model]
        out = self.head(pooled)                  # [B, pred_len]

        if return_embedding:
            return out, pooled
        return out