"""
model.py - PPAD: Patch Predictive Anomaly Detection

Pipeline for each patch i in image x:
  hp = Encoder(crop(x, i))          # isolated patch embedding
  hi = Encoder(mask(x, i))          # context embedding (image with patch i zeroed)
  ho = Predictor(hi, pos_i)         # predicted patch embedding
  score_i = 1 - cosine_sim(hp, ho)  # anomaly score for patch i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Frozen DINOv2 encoder
# ---------------------------------------------------------------------------

class DINOv2Encoder(nn.Module):
    """Wraps a frozen DINOv2 model; returns the CLS token."""

    def __init__(self, model_name: str = 'dinov2_vits14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.embed_dim: int = self.model.embed_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 224, 224] normalized → returns [B, D]"""
        return self.model(x)


# ---------------------------------------------------------------------------
# Transformer predictor
# ---------------------------------------------------------------------------

class PatchPredictor(nn.Module):
    """
    Predicts the embedding of a target patch given:
      - hi  : context embedding [B, D]  (image with that patch masked)
      - pos : patch index       [B]     (integer, learned embedding table)

    The 2-token sequence [context | query] is fed through a small
    TransformerEncoder.  The output at the query position is projected
    to the final predicted embedding ho [B, D].
    """

    def __init__(self, embed_dim: int, num_patches: int,
                 num_heads: int = 6, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(num_patches, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-norm for training stability
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, hi: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        pos = self.pos_embed(patch_idx)          # [B, D]
        seq = torch.stack([hi, pos], dim=1)      # [B, 2, D]
        out = self.transformer(seq)              # [B, 2, D]
        return self.out_proj(out[:, 1])          # [B, D]  ← query token output


# ---------------------------------------------------------------------------
# Full PPAD model
# ---------------------------------------------------------------------------

class PPAD(nn.Module):
    """
    Patch Predictive Anomaly Detection.

    Args:
        patch_grid  : patches per spatial dimension (patch_grid x patch_grid total)
        img_size    : input image size (square)
        encoder_name: DINOv2 variant (dinov2_vits14 / dinov2_vitb14 / ...)
    """

    def __init__(self, patch_grid: int = 4, img_size: int = 224,
                 encoder_name: str = 'dinov2_vits14'):
        super().__init__()
        self.patch_grid  = patch_grid
        self.img_size    = img_size
        self.num_patches = patch_grid * patch_grid
        self.patch_size  = img_size // patch_grid   # pixels per patch side

        self.encoder   = DINOv2Encoder(encoder_name)
        self.predictor = PatchPredictor(
            embed_dim   = self.encoder.embed_dim,
            num_patches = self.num_patches,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patch_coords(self, idx: int):
        """Return (r0, r1, c0, c1) pixel slice for patch index idx."""
        ps = self.patch_size
        r  = idx // self.patch_grid
        c  = idx  % self.patch_grid
        return r * ps, (r + 1) * ps, c * ps, (c + 1) * ps

    def _encode_all_patches(self, images: torch.Tensor) -> torch.Tensor:
        """
        Crop every patch from every image, resize to img_size, encode.
        images : [B, 3, H, W]
        returns: [B, N, D]
        """
        B, N = images.shape[0], self.num_patches
        crops = []
        for idx in range(N):
            r0, r1, c0, c1 = self._patch_coords(idx)
            p = images[:, :, r0:r1, c0:c1]                         # [B, 3, ps, ps]
            p = F.interpolate(p, self.img_size, mode='bilinear', align_corners=False)
            crops.append(p)                                          # [B, 3, H, W]

        # Stack → [B*N, 3, H, W] for one batched encoder call
        crops_cat = torch.cat(crops, dim=0)                         # [B*N, 3, H, W]
        hp_cat    = self.encoder(crops_cat)                         # [B*N, D]
        return hp_cat.view(N, B, -1).permute(1, 0, 2)              # [B, N, D]

    def _encode_all_contexts(self, images: torch.Tensor) -> torch.Tensor:
        """
        For each patch position, zero it out and encode the resulting image.
        images : [B, 3, H, W]
        returns: [B, N, D]
        """
        B, N = images.shape[0], self.num_patches
        masked_list = []
        for idx in range(N):
            r0, r1, c0, c1 = self._patch_coords(idx)
            m = images.clone()
            m[:, :, r0:r1, c0:c1] = 0.0   # 0 ≈ ImageNet mean after normalization
            masked_list.append(m)           # [B, 3, H, W]

        masked_cat = torch.cat(masked_list, dim=0)                  # [B*N, 3, H, W]
        hi_cat     = self.encoder(masked_cat)                       # [B*N, D]
        return hi_cat.view(N, B, -1).permute(1, 0, 2)              # [B, N, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor):
        """
        images: [B, 3, H, W] (normalized)
        Returns patch_scores [B, N] — higher = more anomalous.
        Only the predictor is differentiable; encoder is frozen.
        """
        B, N = images.shape[0], self.num_patches

        hp = self._encode_all_patches(images)    # [B, N, D]  — no grad
        hi = self._encode_all_contexts(images)   # [B, N, D]  — no grad

        # Flatten spatial dims for predictor
        hi_flat  = hi.reshape(B * N, -1)                                # [B*N, D]
        idx_flat = torch.arange(N, device=images.device).repeat(B)      # [B*N]
        ho_flat  = self.predictor(hi_flat, idx_flat)                    # [B*N, D]

        ho = ho_flat.view(B, N, -1)                                     # [B, N, D]

        # 1 - cosine similarity as anomaly score
        scores = 1.0 - F.cosine_similarity(hp, ho, dim=-1)             # [B, N]
        return scores

    def get_anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Upsample patch scores to a per-pixel heatmap.
        images: [B, 3, H, W]
        Returns: [B, 1, H, W]
        """
        scores = self.forward(images)                                   # [B, N]
        H, W   = images.shape[2], images.shape[3]
        grid   = scores.view(-1, 1, self.patch_grid, self.patch_grid)  # [B, 1, g, g]
        return F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False)
