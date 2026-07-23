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
    """Wraps a frozen DINOv2 model; returns the CLS token of the last layer."""

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
    Predicts the mean (mu) and log-variance (logvar) of a target patch given:
      - hi  : context embedding [B, D]  (image with that patch masked)
      - pos : patch index       [B]     (integer, learned embedding table)

    The 2-token sequence [context | query] is fed through a small
    TransformerEncoder. The output at the query position is projected
    to mu [B, D] and logvar [B, D].
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
        self.mu_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.logvar_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, hi: torch.Tensor, patch_idx: torch.Tensor):
        pos = self.pos_embed(patch_idx)          # [B, D]
        seq = torch.stack([hi, pos], dim=1)      # [B, 2, D]
        out = self.transformer(seq)              # [B, 2, D]
        query_out = out[:, 1]                    # [B, D]  ← query token output
        mu = self.mu_proj(query_out)             # [B, D]
        logvar = self.logvar_proj(query_out)     # [B, D]
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar


# ---------------------------------------------------------------------------
# Full PPAD model
# ---------------------------------------------------------------------------

class PPAD(nn.Module):
    """
    Patch Predictive Anomaly Detection (Single 8x8 Grid, Last Layer only).

    Args:
        patch_grids : ignored, kept for signature compatibility
        img_size    : input image size (square)
        encoder_name: DINOv2 variant (dinov2_vits14 / dinov2_vitb14 / ...)
        layers      : ignored, kept for signature compatibility
    """

    def __init__(self, patch_grids: list = None, img_size: int = 224,
                 encoder_name: str = 'dinov2_vits14', layers: list = None, predictor_layers: int = 2):
        super().__init__()
        self.patch_grids = [4,8,12]
        self.img_size    = img_size
        self.layers      = [11]

        self.encoder   = DINOv2Encoder(encoder_name)
        self.predictors = nn.ModuleDict({
            '8': PatchPredictor(
                embed_dim   = self.encoder.embed_dim,
                num_patches = 64,
                num_layers  = predictor_layers,
            )
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
        Crop every patch from every image, resize to img_size, encode.
        images : [B, 3, H, W]
        returns: [B, N, D]
        """
        B, N = images.shape[0], g * g
        crops = []
        for idx in range(N):
            r0, r1, c0, c1 = self._patch_coords(idx, g)
            p = images[:, :, r0:r1, c0:c1]                         # [B, 3, ps, ps]
            p = F.interpolate(p, self.img_size, mode='bilinear', align_corners=False)
            crops.append(p)                                          # [B, 3, H, W]

        # Stack → [B*N, 3, H, W]
        crops_cat = torch.cat(crops, dim=0)                         # [B*N, 3, H, W]
        
        # Encode in chunks to prevent CUDA OOM
        max_batch_size = 64
        hp_list = []
        for i in range(0, crops_cat.shape[0], max_batch_size):
            chunk = crops_cat[i:i + max_batch_size]
            hp_list.append(self.encoder(chunk))                     # [B_chunk, D]
        hp_cat = torch.cat(hp_list, dim=0)                          # [B*N, D]
        return hp_cat.view(N, B, -1).permute(1, 0, 2)               # [B, N, D]

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
            hi_list.append(self.encoder(chunk))                     # [B_chunk, D]
        hi_cat = torch.cat(hi_list, dim=0)                          # [B*N, D]
        return hi_cat.view(N, B, -1).permute(1, 0, 2)               # [B, N, D]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor):
        """
        images: [B, 3, H, W] (normalized)

        If self.training is True:
            Returns a dict {g: (hp_g, mu_g, logvar_g)} for each grid size g in self.patch_grids
        If self.training is False:
            Returns a dict containing:
                - 'fused': average fused pixel-level anomaly map [B, 1, H, W]
                - g (int): individual scale scores [B, N_g]
        """
        B = images.shape[0]
        device = images.device
        g = 8
        N = 64

        if self.training:
            outputs = {}
            with torch.no_grad():
                hp = self._encode_all_patches(images, g)    # [B, N, D]
                hi = self._encode_all_contexts(images, g)   # [B, N, D]

            hi_flat  = hi.reshape(B * N, -1)                                # [B*N, D]
            idx_flat = torch.arange(N, device=device).repeat(B)      # [B*N]
            mu_flat, logvar_flat = self.predictors['8'](hi_flat, idx_flat) # [B*N, D], [B*N, D]
            mu = mu_flat.view(B, N, -1)                                     # [B, N, D]
            logvar = logvar_flat.view(B, N, -1)                             # [B, N, D]
            outputs[g] = (hp, mu, logvar)
            return outputs
        else:
            outputs = {}
            H, W = images.shape[2], images.shape[3]
            with torch.no_grad():
                hp = self._encode_all_patches(images, g)    # [B, N, D]
                hi = self._encode_all_contexts(images, g)   # [B, N, D]

                hi_flat  = hi.reshape(B * N, -1)
                idx_flat = torch.arange(N, device=device).repeat(B)
                mu_flat, logvar_flat = self.predictors['8'](hi_flat, idx_flat)
                mu = mu_flat.view(B, N, -1)
                logvar = logvar_flat.view(B, N, -1)
                
                # Z-score squared anomaly score: (hp - mu)^2 / std^2
                scores = ((hp - mu) ** 2 / torch.exp(logvar)).mean(dim=-1)     # [B, N]
                outputs[g] = scores

                # Upsample to pixel-level heatmap
                grid = scores.view(B, 1, g, g)  # [B, 1, g, g]
                heatmap = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False) # [B, 1, H, W]

            outputs['fused'] = heatmap
            return outputs

    # ------------------------------------------------------------------
    # Checkerboard masking helpers
    # ------------------------------------------------------------------

    def _checkerboard_indices(self, g: int):
        """
        Split the g×g patch grid into two checkerboard halves.
        Returns (black_indices, white_indices) where:
          black = patches at (r, c) with (r + c) % 2 == 0
          white = patches at (r, c) with (r + c) % 2 == 1
        """
        black, white = [], []
        for r in range(g):
            for c in range(g):
                idx = r * g + c
                if (r + c) % 2 == 0:
                    black.append(idx)
                else:
                    white.append(idx)
        return black, white

    def _encode_patches_subset(self, images: torch.Tensor, g: int,
                               indices: list) -> torch.Tensor:
        """
        Crop and encode only the patches at the given indices.
        images : [B, 3, H, W]
        indices: list of patch indices to encode
        returns: [B, len(indices), D]
        """
        B = images.shape[0]
        K = len(indices)
        crops = []
        for idx in indices:
            r0, r1, c0, c1 = self._patch_coords(idx, g)
            p = images[:, :, r0:r1, c0:c1]
            p = F.interpolate(p, self.img_size, mode='bilinear', align_corners=False)
            crops.append(p)

        crops_cat = torch.cat(crops, dim=0)                         # [B*K, 3, H, W]

        max_batch_size = 64
        hp_list = []
        for i in range(0, crops_cat.shape[0], max_batch_size):
            chunk = crops_cat[i:i + max_batch_size]
            hp_list.append(self.encoder(chunk))
        hp_cat = torch.cat(hp_list, dim=0)                          # [B*K, D]
        return hp_cat.view(K, B, -1).permute(1, 0, 2)               # [B, K, D]

    def _encode_checkerboard_context(self, images: torch.Tensor, g: int,
                                     mask_indices: list) -> torch.Tensor:
        """
        Zero out ALL patches at mask_indices simultaneously, encode once.
        images      : [B, 3, H, W]
        mask_indices: list of patch indices to zero out
        returns     : [B, D]  (single context embedding per image)
        """
        m = images.clone()
        for idx in mask_indices:
            r0, r1, c0, c1 = self._patch_coords(idx, g)
            m[:, :, r0:r1, c0:c1] = 0.0
        return self.encoder(m)                                      # [B, D]

    # ------------------------------------------------------------------
    # Checkerboard forward (eval only)
    # ------------------------------------------------------------------

    def forward_checkerboard(self, images: torch.Tensor):
        """
        Checkerboard evaluation: mask n/2 patches at once in two passes.

        Pass 1: mask "black" patches (r+c even), predict their embeddings
        Pass 2: mask "white" patches (r+c odd),  predict their embeddings

        Returns the same dict format as forward() in eval mode:
          - 'fused': [B, 1, H, W] averaged anomaly heatmap
          - g (int): [B, N] per-patch scores for each grid scale
        """
        assert not self.training, "forward_checkerboard is eval-only"
        B = images.shape[0]
        device = images.device
        H, W = images.shape[2], images.shape[3]
        g = 8
        N = 64

        black_idx, white_idx = self._checkerboard_indices(g)

        # Full score tensor to fill
        all_scores = torch.zeros(B, N, device=device)

        with torch.no_grad():
            for mask_indices, predict_indices in [
                (black_idx, black_idx),   # Pass 1: mask black, predict black
                (white_idx, white_idx),   # Pass 2: mask white, predict white
            ]:
                K = len(predict_indices)

                # 1. Encode isolated crops for the patches we want to predict
                hp = self._encode_patches_subset(images, g, predict_indices)  # [B, K, D]

                # 2. Encode context (all masked patches zeroed simultaneously)
                hi = self._encode_checkerboard_context(images, g, mask_indices)  # [B, D]

                # 3. Predict embeddings for each masked patch
                hi_expanded = hi.unsqueeze(1).expand(B, K, -1)  # [B, K, D]
                hi_flat = hi_expanded.reshape(B * K, -1)        # [B*K, D]

                # Patch position indices
                idx_tensor = torch.tensor(predict_indices, device=device)  # [K]
                idx_flat = idx_tensor.unsqueeze(0).expand(B, -1).reshape(B * K)  # [B*K]

                mu_flat, logvar_flat = self.predictors['8'](hi_flat, idx_flat)  # [B*K, D]
                mu = mu_flat.view(B, K, -1)                          # [B, K, D]
                logvar = logvar_flat.view(B, K, -1)                  # [B, K, D]

                scores = ((hp - mu) ** 2 / torch.exp(logvar)).mean(dim=-1)   # [B, K]

                # Scatter into full score tensor
                idx_tensor = torch.tensor(predict_indices, device=device)
                all_scores[:, idx_tensor] = scores

        outputs = {}
        outputs[g] = all_scores  # [B, N]

        # Upsample to pixel-level heatmap
        grid = all_scores.view(B, 1, g, g)
        heatmap = F.interpolate(grid, size=(H, W), mode='bilinear', align_corners=False)
        outputs['fused'] = heatmap
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

