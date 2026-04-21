import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, patch_len, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len * input_dim, d_model)

    def forward(self, x):
        B, L, C = x.shape

        num_patches = L // self.patch_len

        x = x[:, :num_patches * self.patch_len, :]
        x = x.reshape(B, num_patches, self.patch_len * C)

        return self.proj(x)


class RegimeAwarePatchTST(nn.Module):

    def __init__(
        self,
        input_dim,
        input_len=336,
        pred_len=96,
        patch_len=16,
        d_model=64,
        n_heads=4,
        num_layers=2,
        dropout=0.2,
        num_regimes=3,
        regime_dim=16
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            input_dim,
            patch_len,
            d_model
        )

        num_patches = input_len // patch_len

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regime_embedding = nn.Embedding(
            num_regimes,
            regime_dim
        )

        combined_dim = d_model + regime_dim

        self.gate = nn.Sequential(

            nn.Linear(combined_dim, combined_dim),

            nn.ReLU(),

            nn.Linear(combined_dim, combined_dim),

            nn.Sigmoid()

        )

        self.head = nn.Sequential(

            nn.LayerNorm(combined_dim),

            nn.Linear(combined_dim, pred_len)

        )

    def forward(self, x, regime_labels):

        x = self.patch_embed(x)

        x = x + self.pos_embedding[:, :x.size(1), :]

        h = self.encoder(x)

        temporal_embedding = h.mean(dim=1)

        regime_embedding = self.regime_embedding(regime_labels)

        combined = torch.cat(

            [

                temporal_embedding,

                regime_embedding

            ],

            dim=-1

        )

        gated = combined * self.gate(combined)

        out = self.head(gated)

        return out