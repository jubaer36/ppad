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
    Patch Predictive Anomaly Detection (Multi-scale Grid).

    Args:
        patch_grids : list of patch grid sizes (e.g. [4, 8, 16])
        img_size    : input image size (square)
        encoder_name: DINOv2 variant (dinov2_vits14 / dinov2_vitb14 / ...)
    """

    def __init__(self, patch_grids: list = [4, 8, 16], img_size: int = 224,
                 encoder_name: str = 'dinov2_vits14'):
        super().__init__()
        self.patch_grids = patch_grids
        self.img_size    = img_size

        self.encoder   = DINOv2Encoder(encoder_name)
        self.predictors = nn.ModuleDict({
            str(g): PatchPredictor(
                embed_dim   = self.encoder.embed_dim,
                num_patches = g * g,
            )
            for g in patch_grids
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patch_coords(self, idx: int, g: int):
        """Return (r0, r1, c0, c1) pixel slice for patch index idx at grid scale g."""
        ps = self.img_size // g
        w  = self.img_size
        r  = idx // g
        c  = idx  % g

        if g > 1:
            r0 = int(r * (w - ps) / (g - 1))
            c0 = int(c * (w - ps) / (g - 1))
        else:
            r0 = 0
            c0 = 0

        return r0, r0 + ps, c0, c0 + ps

    def _encode_all_patches(self, images: torch.Tensor, g: int) -> torch.Tensor:
        """
        Crop every patch from every image and encode directly at native crop size.
        images : [B, 3, H, W]
        returns: [B, N, D]
        """
        B, N = images.shape[0], g * g
        crops = []
        for idx in range(N):
            r0, r1, c0, c1 = self._patch_coords(idx, g)
            p = images[:, :, r0:r1, c0:c1]                         # [B, 3, ps, ps]
            crops.append(p)                                          # [B, 3, ps, ps]

        # Stack → [B*N, 3, ps, ps]
        crops_cat = torch.cat(crops, dim=0)                         # [B*N, 3, ps, ps]
        
        # Encode in chunks to prevent CUDA OOM
        max_batch_size = 64
        hp_list = []
        for i in range(0, crops_cat.shape[0], max_batch_size):
            chunk = crops_cat[i:i + max_batch_size]
            hp_list.append(self.encoder(chunk))
        hp_cat = torch.cat(hp_list, dim=0)                          # [B*N, D]
        return hp_cat.view(N, B, -1).permute(1, 0, 2)              # [B, N, D]

    def _encode_all_contexts(self, images: torch.Tensor, g: int) -> torch.Tensor:
        """
        For each patch position, zero it out and encode the resulting image.
        images : [B, 3, H, W]
        returns: [B, N, D]
        """
        B, N = images.shape[0], g * g
        masked_list = []
        for idx in range(N):
            r0, r1, c0, c1 = self._patch_coords(idx, g)
            m = images.clone()
            m[:, :, r0:r1, c0:c1] = 0.0   # 0 ≈ ImageNet mean after normalization
            masked_list.append(m)           # [B, 3, H, W]

        # Stack → [B*N, 3, H, W]
        masked_cat = torch.cat(masked_list, dim=0)                  # [B*N, 3, H, W]
        
        # Encode in chunks to prevent CUDA OOM
        max_batch_size = 64
        hi_list = []
        for i in range(0, masked_cat.shape[0], max_batch_size):
            chunk = masked_cat[i:i + max_batch_size]
            hi_list.append(self.encoder(chunk))
        hi_cat = torch.cat(hi_list, dim=0)                          # [B*N, D]
        return hi_cat.view(N, B, -1).permute(1, 0, 2)              # [B, N, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor):
        """
        images: [B, 3, H, W] (normalized)

        If self.training is True:
            Returns a dict {g: (hp_g, ho_g)} for each grid size g in self.patch_grids
        If self.training is False:
            Returns a dict containing:
                - 'fused': average fused pixel-level anomaly map [B, 1, H, W]
                - g (int): individual scale scores [B, N_g]
        """
        B = images.shape[0]
        device = images.device

        if self.training:
            outputs = {}
            for g in self.patch_grids:
                N = g * g
                with torch.no_grad():
                    hp = self._encode_all_patches(images, g)    # [B, N, D]
                    hi = self._encode_all_contexts(images, g)   # [B, N, D]

                # Flatten spatial dims for predictor
                hi_flat  = hi.reshape(B * N, -1)                                # [B*N, D]
                idx_flat = torch.arange(N, device=device).repeat(B)      # [B*N]
                ho_flat  = self.predictors[str(g)](hi_flat, idx_flat)           # [B*N, D]

                ho = ho_flat.view(B, N, -1)                                     # [B, N, D]
                outputs[g] = (hp, ho)
            return outputs
        else:
            outputs = {}
            heatmaps = []
            H, W = images.shape[2], images.shape[3]

            for g in self.patch_grids:
                N = g * g
                with torch.no_grad():
                    hp = self._encode_all_patches(images, g)    # [B, N, D]
                    hi = self._encode_all_contexts(images, g)   # [B, N, D]

                    hi_flat  = hi.reshape(B * N, -1)
                    idx_flat = torch.arange(N, device=device).repeat(B)
                    ho_flat  = self.predictors[str(g)](hi_flat, idx_flat)

                    ho = ho_flat.view(B, N, -1)
                    scores = 1.0 - F.cosine_similarity(hp, ho, dim=-1)             # [B, N]
                    outputs[g] = scores

                    # Upsample to pixel-level heatmap
                    grid = scores.view(B, 1, g, g)  # [B, 1, g, g]
                    heatmap_g = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False) # [B, 1, H, W]
                    heatmaps.append(heatmap_g)

            # Average (fuse) the upsampled anomaly maps
            fused_heatmap = torch.stack(heatmaps, dim=0).mean(dim=0)  # [B, 1, H, W]
            outputs['fused'] = fused_heatmap
            return outputs

    def get_anomaly_map(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns the fused per-pixel anomaly map.
        images: [B, 3, H, W]
        Returns: [B, 1, H, W]
        """
        was_training = self.training
        self.eval()
        outputs = self.forward(images)
        if was_training:
            self.train()
        return outputs['fused']

