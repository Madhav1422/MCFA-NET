# -*- coding: windows-1252 -*-
# -*- coding: utf-8 -*-
"""
MCFA_Net_main.py
================
MCFA-Net: Multi-scale Cross-domain Frequency-Aware Network
===========================================================

NOVELTY STATEMENT
-----------------
MCFA-Net introduces THREE synergistic novel components not seen together
in any prior brain-tumor classification work:

1. **Dual-Domain Attention Fusion (DDAF)**
   - Operates simultaneously in SPATIAL domain (EfficientNet-B3 features)
     and FREQUENCY domain (Wavelet + FFT decomposed features).
   - A learnable cross-domain attention gate DYNAMICALLY WEIGHTS how much
     spatial vs. frequency evidence to trust per sample.
   - This is fundamentally different from SV-IBN, which only gates inside
     the frequency domain with a fixed SE-channel attention.

2. **Hierarchical Prototype Contrastive Bottleneck (HPCB)**
   - Maintains per-class learnable PROTOTYPE vectors.
   - Computes prototype-distance-aware contrastive loss so the latent
     space is explicitly structured: intra-class tight, inter-class spread.
   - A variational sampling head draws from N(mu, sigma) conditioned on
     prototype proximity — combining VIB with metric learning.
   - SV-IBN uses a plain VIB with no prototype structure.

3. **Adaptive Multi-scale Pyramid Pooling (AMPP)**
   - Extracts features at 4 spatial scales (1x1, 2x2, 3x3, 6x6) then
     fuses them with a self-attention weighting that is INPUT-DEPENDENT.
   - This replaces fixed global-average pooling used in all baselines and
     in SV-IBN, adapting pooling scale to tumour size/location.

Architecture:
  Input ? EfficientNet-B3 (multi-scale feature maps) ? DDAF(freq+spatial)
        ? AMPP (adaptive pooling) ? HPCB (proto-variational bottleneck)
        ? Classifier

Models trained:
  - EfficientNet_B3                : baseline
  - MCFA_no_DDAF                   : ablation A (AMPP + HPCB only)
  - MCFA_no_HPCB                   : ablation B (DDAF + AMPP only)
  - MCFA_no_AMPP                   : ablation C (DDAF + HPCB only)
  - MCFA_Net_EfficientNet_B3       : PROPOSED (full model)

Seeds: 42–47  (n=6) for Wilcoxon + McNemar statistical validity.

Outputs
-------
MCFA_Net_RESULTS_2026/
  EfficientNet_B3/            metrics/ curves/ gradcam/
  MCFA_no_DDAF/               metrics/ curves/ gradcam/
  MCFA_no_HPCB/               metrics/ curves/ gradcam/
  MCFA_no_AMPP/               metrics/ curves/ gradcam/
  MCFA_Net_EfficientNet_B3/   metrics/ curves/ gradcam/
  all_results.csv
  summary_aggregated.csv
  ablation_summary.csv
  wilcoxon_n6.csv
  mcnemar_pooled.csv
  superiority_table.csv
  ablation_superiority.csv
  boxplot_all_metrics.png
  barplot_{metric}.png  (x5)
  mcnemar_pvalue_heatmap.png
  cohens_d_heatmap.png
"""

import os, gc, random, itertools, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.stats import chi2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms.v2 as v2

import timm
import pywt                          # pip install PyWavelets
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    roc_curve, auc, precision_score, recall_score,
    classification_report
)
from sklearn.model_selection import train_test_split
from scipy import stats

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Bold / publication-quality matplotlib defaults
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.weight":         "bold",
    "axes.labelweight":    "bold",
    "axes.titleweight":    "bold",
    "axes.titlesize":      14,
    "axes.labelsize":      12,
    "xtick.labelsize":     11,
    "ytick.labelsize":     11,
    "legend.fontsize":     10,
    "figure.titlesize":    15,
    "figure.titleweight":  "bold",
    "lines.linewidth":     2.2,
    "lines.markersize":    7,
    "savefig.dpi":         200,
    "savefig.bbox":        "tight",
})


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TRAIN_PATH   ="/nfsshare/users/raghavan/brainzz/Brain tumor dataset/Training/"
TEST_PATH    = "/nfsshare/users/raghavan/brainzz/Brain tumor dataset/Test/"
SAVE_DIR     = Path("MCFA_Net_RESULTS_2026_final_imp")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE       = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 224
LR           = 1e-4
EPOCHS       = 60
PATIENCE     = 12
LATENT_DIM   = 512
PROTO_DIM    = 256          # prototype embedding dim
BETA_KL_MAX  = 1e-3         # KL weight (annealed)
BETA_PROTO   = 0.5          # prototype contrastive loss weight
LABEL_SMOOTH = 0.10
MIXUP_ALPHA  = 0.4
ALPHA        = 0.05         # significance level

SEEDS  = [42, 43, 44, 45, 46, 47]

MODELS = [
    "EfficientNet_B3",               # baseline (no novelty)
    "MCFA_no_DDAF",                  # ablation A: AMPP + HPCB, no freq gate
    "MCFA_no_HPCB",                  # ablation B: DDAF + AMPP, no prototypes
    "MCFA_no_AMPP",                  # ablation C: DDAF + HPCB, fixed pooling
    "MCFA_Net_EfficientNet_B3",      # PROPOSED: full MCFA-Net
]

MODEL_BATCH = {m: 16 for m in MODELS}

ABLATION_ROLES = {
    "EfficientNet_B3":
        "Baseline. Pretrained EfficientNet-B3 with label smoothing. "
        "No DDAF, no HPCB, no AMPP.",
    "MCFA_no_DDAF":
        "Ablation A. AMPP + HPCB only. "
        "Dual-Domain Attention Fusion removed; isolates AMPP+HPCB contribution.",
    "MCFA_no_HPCB":
        "Ablation B. DDAF + AMPP only. "
        "Hierarchical Prototype Contrastive Bottleneck removed; "
        "plain global-average pooling bottleneck used instead.",
    "MCFA_no_AMPP":
        "Ablation C. DDAF + HPCB only. "
        "Adaptive Multi-scale Pyramid Pooling replaced with fixed GAP; "
        "isolates DDAF+HPCB contribution.",
    "MCFA_Net_EfficientNet_B3":
        "PROPOSED. Full MCFA-Net: DDAF + AMPP + HPCB + EfficientNet-B3. "
        "All three novel components active simultaneously.",
}

METRICS = ["acc", "f1_macro", "precision_macro", "recall_macro", "auc_macro"]
METRIC_LABELS = {
    "acc":             "Accuracy",
    "f1_macro":        "Macro F1",
    "precision_macro": "Macro Precision",
    "recall_macro":    "Macro Recall",
    "auc_macro":       "Macro AUC",
}


# -----------------------------------------------------------------------------
# Directory helpers
# -----------------------------------------------------------------------------
def get_model_dirs(model_name: str) -> dict:
    root    = SAVE_DIR / model_name
    metrics = root / "metrics"
    curves  = root / "curves"
    gradcam = root / "gradcam"
    for d in [root, metrics, curves, gradcam]:
        d.mkdir(parents=True, exist_ok=True)
    note = root / "ablation_note.txt"
    if not note.exists():
        note.write_text(
            f"Model : {model_name}\n"
            f"Role  : {ABLATION_ROLES.get(model_name, 'Unknown')}\n"
        )
    return {"root": root, "metrics": metrics, "curves": curves, "gradcam": gradcam}


def get_model_paths(model_name: str, seed: int, dirs: dict) -> dict:
    return {
        "test_metrics": dirs["metrics"] / f"test_metrics_seed{seed}.csv",
        "cm_csv":       dirs["metrics"] / f"cm_seed{seed}.csv",
        "cm_png":       dirs["metrics"] / f"cm_seed{seed}.png",
        "roc_data":     dirs["metrics"] / f"roc_data_seed{seed}.csv",
        "roc_png":      dirs["metrics"] / f"roc_seed{seed}.png",
        "history_csv":  dirs["metrics"] / f"history_seed{seed}.csv",
        "curves_png":   dirs["curves"]  / f"curves_seed{seed}.png",
        "gradcam_base": dirs["gradcam"],
    }


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# -----------------------------------------------------------------------------
# MixUp
# -----------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.4):
    lam     = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx     = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# -----------------------------------------------------------------------------
# NOVEL COMPONENT 1: Dual-Domain Attention Fusion (DDAF)
# -----------------------------------------------------------------------------
class WaveletFFTExtractor(nn.Module):
    """
    Extracts a hybrid frequency representation:
      - 2D DWT (Haar) ? LL / LH / HL / HH sub-bands
      - FFT magnitude spectrum
    Both are projected to `out_channels` and concatenated.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Project 4 wavelet sub-bands (each in_channels) to out_channels
        self.wavelet_proj = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        # Project FFT magnitude to out_channels
        self.fft_proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        # Fuse wavelet + FFT ? out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def _dwt2d(self, x: torch.Tensor) -> torch.Tensor:
        """Haar DWT via average / difference filters (differentiable)."""
        B, C, H, W = x.shape
        # Pad to even spatial dims
        x = F.pad(x, (0, W % 2, 0, H % 2))
        h2 = x.shape[-2] // 2
        w2 = x.shape[-1] // 2
        # 2×2 non-overlapping windows
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        LL = (x00 + x01 + x10 + x11) * 0.25
        LH = (x00 - x01 + x10 - x11) * 0.25
        HL = (x00 + x01 - x10 - x11) * 0.25
        HH = (x00 - x01 - x10 + x11) * 0.25
        return torch.cat([LL, LH, HL, HH], dim=1)   # (B, 4C, H/2, W/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Wavelet branch ---
        wt = self._dwt2d(x)
        wt = F.interpolate(wt, size=x.shape[-2:], mode='bilinear', align_corners=False)
        wt = self.wavelet_proj(wt)

        # --- FFT branch ---
        fft_mag = torch.fft.rfft2(x, norm='ortho').abs()
        # rfft2 output shape: (B, C, H, W//2+1) ? resize back
        fft_mag = F.interpolate(fft_mag, size=x.shape[-2:], mode='bilinear', align_corners=False)
        fft_feat = self.fft_proj(fft_mag)

        return self.fuse(torch.cat([wt, fft_feat], dim=1))


class DualDomainAttentionFusion(nn.Module):
    """
    NOVEL COMPONENT 1 – DDAF
    Learns a PER-SAMPLE gate a ? [0,1] that mixes spatial and frequency
    feature maps:  out = a * spatial + (1-a) * frequency
    Gate is computed from the concatenation of BOTH streams, so it is
    input-content-aware (different tumour types ? different a).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.freq_extractor = WaveletFFTExtractor(channels, channels)
        # Squeeze ? gate score
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels * 2),
            nn.Linear(channels * 2, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid(),
        )
        # Lightweight refinement conv after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.freq_extractor(x)
        # gate: (B, 1)
        alpha = self.gate_net(torch.cat([
            x.mean(dim=(2, 3)),
            freq.mean(dim=(2, 3))
        ], dim=1)).unsqueeze(-1).unsqueeze(-1)   # (B,1,1,1)
        fused = alpha * x + (1.0 - alpha) * freq
        return self.refine(fused)


# -----------------------------------------------------------------------------
# NOVEL COMPONENT 2: Adaptive Multi-scale Pyramid Pooling (AMPP)
# -----------------------------------------------------------------------------
class AdaptiveMultiScalePyramidPooling(nn.Module):
    """
    NOVEL COMPONENT 2 – AMPP
    Pools feature maps at 4 spatial scales [1×1, 2×2, 3×3, 6×6].
    A small self-attention mechanism computes INPUT-DEPENDENT weights
    over the 4 scales, so the network dynamically chooses which spatial
    resolution to trust for each image.
    Output: flat vector of size `in_channels`.
    """
    SCALES = [1, 2, 3, 6]

    def __init__(self, in_channels: int):
        super().__init__()
        n = len(self.SCALES)
        # Each scale is projected to in_channels//n, then concatenated
        self.branch_proj = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Flatten(),
                nn.Linear(in_channels * s * s, in_channels // n),
                nn.LayerNorm(in_channels // n),
                nn.GELU(),
            )
            for s in self.SCALES
        ])
        # Self-attention scale weighting: (B, n) ? softmax
        self.scale_attn = nn.Sequential(
            nn.Linear(in_channels, n),
            nn.Softmax(dim=-1),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fast global descriptor for attention query
        gap = x.mean(dim=(2, 3))                               # (B, C)
        scale_w = self.scale_attn(gap)                         # (B, 4)

        branches = []
        for bp in self.branch_proj:
            branches.append(bp(x))                             # each (B, C//4)

        # Weighted combination
        stacked = torch.stack(branches, dim=1)                 # (B, 4, C//4)
        # expand weights: (B, 4, 1)
        weighted = (stacked * scale_w.unsqueeze(-1)).sum(dim=1)  # (B, C//4)

        # Concat all branches + weighted sum for richness
        concat = torch.cat(branches, dim=1)                    # (B, C)
        out    = concat + self.out_proj(concat)                # residual
        return out


# -----------------------------------------------------------------------------
# NOVEL COMPONENT 3: Hierarchical Prototype Contrastive Bottleneck (HPCB)
# -----------------------------------------------------------------------------
class HierarchicalPrototypeContrastiveBottleneck(nn.Module):
    """
    NOVEL COMPONENT 3 – HPCB
    - Maintains K learnable per-class prototype vectors p_k ? R^{proto_dim}.
    - Projects input to prototype space, computes softmax similarity.
    - Reparameterizes: mu, logvar conditioned on prototype-weighted context.
    - Returns z (sample), mu, logvar, and a contrastive prototype loss.

    Prototype contrastive loss:
      L_proto = -log [ exp(-d(z,p_y)) / S_k exp(-d(z,p_k)) ]
    where d = squared Euclidean distance.  This pulls z toward the correct
    prototype and pushes it away from all others simultaneously.
    """
    def __init__(self, in_dim: int, proto_dim: int, latent_dim: int,
                 num_classes: int):
        super().__init__()
        self.proto_dim   = proto_dim
        self.latent_dim  = latent_dim
        self.num_classes = num_classes

        # Learnable class prototypes
        self.prototypes = nn.Parameter(
            torch.randn(num_classes, proto_dim) * 0.1
        )

        # Project input to prototype space
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, proto_dim),
            nn.LayerNorm(proto_dim),
            nn.GELU(),
        )

        # Proto-conditioned mu / logvar
        # Input = [z_enc || proto_context] of size proto_dim * 2
        self.mu_head     = nn.Linear(proto_dim * 2, latent_dim)
        self.logvar_head = nn.Linear(proto_dim * 2, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar.clamp(-4, 4))
        return mu + torch.randn_like(std) * std

    def prototype_contrastive_loss(self, z_enc: torch.Tensor,
                                    labels: torch.Tensor) -> torch.Tensor:
        # z_enc: (B, proto_dim)   labels: (B,)
        # distances to each prototype: (B, K)
        diff  = z_enc.unsqueeze(1) - self.prototypes.unsqueeze(0)  # (B,K,D)
        dists = (diff ** 2).sum(-1)                                  # (B, K)
        # Numerator: distance to correct prototype
        pos_d = dists[torch.arange(len(labels)), labels]            # (B,)
        # Log-softmax style contrastive
        log_probs = -pos_d - torch.logsumexp(-dists, dim=1)         # (B,)
        return -log_probs.mean()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        z_enc = self.encoder(x)                                     # (B, proto_dim)

        # Prototype similarities ? soft prototype context
        diff    = z_enc.unsqueeze(1) - self.prototypes.unsqueeze(0) # (B,K,D)
        dists   = (diff ** 2).sum(-1)                               # (B, K)
        sim_w   = F.softmax(-dists, dim=1)                          # (B, K)
        context = torch.einsum('bk,kd->bd', sim_w, self.prototypes) # (B, proto_dim)

        # Conditioned variational head
        h      = torch.cat([z_enc, context], dim=1)                 # (B, 2*proto_dim)
        mu     = self.mu_head(h)
        logvar = self.logvar_head(h)
        z      = self.reparameterize(mu, logvar)

        # Prototype contrastive loss (only during training with labels)
        proto_loss = torch.tensor(0.0, device=x.device)
        if labels is not None and self.training:
            proto_loss = self.prototype_contrastive_loss(z_enc, labels)

        return z, mu, logvar, proto_loss


# -----------------------------------------------------------------------------
# Model Factory
# -----------------------------------------------------------------------------
def create_model(name: str, num_classes: int) -> nn.Module:

    # -- Baseline ------------------------------------------------------------
    if name == "EfficientNet_B3":
        base = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, num_classes=num_classes
        )
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.model        = base
                self.target_layer = base.blocks[-1][-1].conv_pwl
            def forward(self, x, labels=None):
                return self.model(x), None, None, torch.tensor(0.0, device=x.device)
        return Wrapper()

    # -- Ablation A: AMPP + HPCB, no DDAF ------------------------------------
    elif name == "MCFA_no_DDAF":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.ampp         = AdaptiveMultiScalePyramidPooling(nf)
                self.hpcb         = HierarchicalPrototypeContrastiveBottleneck(
                                        nf, PROTO_DIM, LATENT_DIM, num_classes)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x, labels=None):
                feats  = self.backbone(x)[-1]
                pooled = self.ampp(feats)
                z, mu, logvar, ploss = self.hpcb(pooled, labels)
                return self.head(self.dropout(z)), mu, logvar, ploss
        return Model()

    # -- Ablation B: DDAF + AMPP, no HPCB ------------------------------------
    elif name == "MCFA_no_HPCB":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.ddaf         = DualDomainAttentionFusion(nf)
                self.ampp         = AdaptiveMultiScalePyramidPooling(nf)
                self.proj         = nn.Sequential(
                                        nn.Linear(nf, LATENT_DIM),
                                        nn.LayerNorm(LATENT_DIM), nn.GELU())
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x, labels=None):
                feats  = self.backbone(x)[-1]
                feats  = self.ddaf(feats)
                pooled = self.ampp(feats)
                z      = self.proj(pooled)
                return self.head(self.dropout(z)), None, None, torch.tensor(0.0, device=x.device)
        return Model()

    # -- Ablation C: DDAF + HPCB, fixed GAP (no AMPP) ------------------------
    elif name == "MCFA_no_AMPP":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.ddaf         = DualDomainAttentionFusion(nf)
                self.hpcb         = HierarchicalPrototypeContrastiveBottleneck(
                                        nf, PROTO_DIM, LATENT_DIM, num_classes)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x, labels=None):
                feats  = self.backbone(x)[-1]
                feats  = self.ddaf(feats)
                pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # fixed GAP
                z, mu, logvar, ploss = self.hpcb(pooled, labels)
                return self.head(self.dropout(z)), mu, logvar, ploss
        return Model()

    # -- PROPOSED: Full MCFA-Net -----------------------------------------------
    elif name == "MCFA_Net_EfficientNet_B3":
        backbone = timm.create_model(
            "efficientnet_b3.ra2_in1k", pretrained=True, features_only=True
        )
        nf = backbone.feature_info[-1]['num_chs']

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone     = backbone
                self.ddaf         = DualDomainAttentionFusion(nf)
                self.ampp         = AdaptiveMultiScalePyramidPooling(nf)
                self.hpcb         = HierarchicalPrototypeContrastiveBottleneck(
                                        nf, PROTO_DIM, LATENT_DIM, num_classes)
                self.dropout      = nn.Dropout(0.3)
                self.head         = nn.Linear(LATENT_DIM, num_classes)
                self.target_layer = backbone.blocks[-1][-1].conv_pwl
            def forward(self, x, labels=None):
                feats  = self.backbone(x)[-1]            # spatial feature map
                feats  = self.ddaf(feats)                 # DDAF: freq ? spatial
                pooled = self.ampp(feats)                 # AMPP: adaptive pooling
                z, mu, logvar, ploss = self.hpcb(pooled, labels)  # HPCB
                return self.head(self.dropout(z)), mu, logvar, ploss
        return Model()

    else:
        raise ValueError(f"Unknown model name: {name}")


# -----------------------------------------------------------------------------
# GradCAM (compatible with all model variants)
# -----------------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None

        def fwd(module, inp, output):
            self.activations = output.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._fh = target_layer.register_forward_hook(fwd)
        self._bh = target_layer.register_full_backward_hook(bwd)

    def generate(self, x: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        logits, _, _, _ = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        logits[:, class_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam     = F.interpolate(cam, size=x.shape[2:], mode='bilinear',
                                align_corners=False)
        cam_min, cam_max = cam.min(), cam.max()
        return ((cam - cam_min) / (cam_max - cam_min + 1e-8)).cpu().numpy()[0, 0]

    def release(self):
        self._fh.remove()
        self._bh.remove()


# -----------------------------------------------------------------------------
# Denormalise helper
# -----------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

def denorm(t: torch.Tensor) -> np.ndarray:
    t = t.cpu() * _STD.view(3, 1, 1) + _MEAN.view(3, 1, 1)
    return (t.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Save test metrics
# -----------------------------------------------------------------------------
def save_all_test_metrics(model_name, seed, classes, y_true, y_pred, y_prob,
                           out_dir=None):
    save_to = Path(out_dir) if out_dir else SAVE_DIR
    rows    = []
    report  = classification_report(
        y_true, y_pred, target_names=classes,
        output_dict=True, zero_division=0
    )
    per_auc = []
    for cls_idx, cls_name in enumerate(classes):
        r           = report[cls_name]
        fpr, tpr, _ = roc_curve((y_true == cls_idx).astype(int), y_prob[:, cls_idx])
        cls_auc     = auc(fpr, tpr)
        per_auc.append(cls_auc)
        cm_bin      = confusion_matrix(
            (y_true == cls_idx).astype(int),
            (y_pred == cls_idx).astype(int)
        )
        tn = cm_bin[0, 0] if cm_bin.shape == (2, 2) else 0
        fp = cm_bin[0, 1] if cm_bin.shape == (2, 2) else 0
        rows.append({
            "model": model_name, "seed": seed, "class": cls_name,
            "precision":   round(r["precision"], 4),
            "recall":      round(r["recall"],    4),
            "f1_score":    round(r["f1-score"],  4),
            "support":     int(r["support"]),
            "auc":         round(cls_auc,         4),
            "specificity": round(tn / (tn + fp + 1e-8), 4),
        })

    acc         = accuracy_score(y_true, y_pred)
    macro_prec  = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    macro_rec   = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
    macro_f1    = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred,        average='weighted', zero_division=0)
    macro_auc   = float(np.mean(per_auc))

    for tag, vals in [
        ("MACRO_AVG",   {"precision": macro_prec, "recall": macro_rec,
                         "f1_score":  macro_f1,   "auc":    macro_auc}),
        ("ACCURACY",    {"f1_score": acc}),
        ("WEIGHTED_F1", {"f1_score": weighted_f1}),
    ]:
        row = {"model": model_name, "seed": seed, "class": tag,
               "precision": "", "recall": "", "f1_score": "",
               "support": int(y_true.shape[0]), "auc": "", "specificity": ""}
        for k, v in vals.items():
            row[k] = round(v, 4)
        rows.append(row)

    out_path = save_to / f"test_metrics_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  [ok]  Saved {out_path.name}")
    return acc, macro_f1, macro_prec, macro_rec, macro_auc


# -----------------------------------------------------------------------------
# GradCAM figure (bold, publication-ready, saved per seed per class)
# -----------------------------------------------------------------------------
def save_gradcam_figure(orig_img, cam, cls_name, pred_name,
                         correct: bool, model_name, seed, img_idx,
                         cam_base: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(orig_img)
    axes[0].set_title("Original Image", fontweight="bold", fontsize=13)
    axes[0].axis('off')

    im = axes[1].imshow(cam, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("GradCAM Heatmap", fontweight="bold", fontsize=13)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(orig_img)
    axes[2].imshow(plt.cm.jet(cam)[:, :, :3], alpha=0.45)
    result_str = "? Correct" if correct else "? Wrong"
    colour     = "green"     if correct else "red"
    axes[2].set_title(
        f"True: {cls_name} | Pred: {pred_name}\n{result_str}",
        fontweight="bold", fontsize=12, color=colour
    )
    axes[2].axis('off')

    fig.suptitle(
        f"{model_name.replace('_', ' ')}  –  Seed {seed}",
        fontweight="bold", fontsize=14, y=1.02
    )
    plt.tight_layout()
    save_path = cam_base / cls_name / f"img_{img_idx:05d}_seed{seed}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
# Single seed: train + evaluate
# -----------------------------------------------------------------------------
def run_one_seed(seed: int, model_name: str) -> dict:

    # Skip if already done
    existing_csv = SAVE_DIR / "all_results.csv"
    if existing_csv.exists():
        ex = pd.read_csv(existing_csv)
        if not ex[(ex["model"] == model_name) & (ex["seed"] == seed)].empty:
            print(f"  ->  {model_name}  seed {seed}  already done — skipping")
            return ex[(ex["model"] == model_name) &
                      (ex["seed"] == seed)].iloc[0].to_dict()

    set_seed(seed)
    dirs  = get_model_dirs(model_name)
    paths = get_model_paths(model_name, seed, dirs)

    print(f"\n{'='*70}")
    print(f"  Seed {seed}  |  Model: {model_name}")
    print(f"{'='*70}")

    batch_size = MODEL_BATCH.get(model_name, 16)
    use_mixup  = (model_name == "MCFA_Net_EfficientNet_B3")
    use_proto  = model_name in ("MCFA_no_DDAF", "MCFA_no_AMPP",
                                 "MCFA_Net_EfficientNet_B3")

    # -- Transforms ----------------------------------------------------------
    train_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.RandomResizedCrop(IMG_SIZE, scale=(0.80, 1.0)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(p=0.1),
        v2.RandomRotation(20),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        v2.RandomGrayscale(p=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = v2.Compose([
        v2.Lambda(lambda img: img.convert("RGB")),
        v2.Resize((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # -- Datasets -------------------------------------------------------------
    full_train   = datasets.ImageFolder(TRAIN_PATH, transform=train_tf)
    full_val_ref = datasets.ImageFolder(TRAIN_PATH, transform=val_tf)
    test_ds      = datasets.ImageFolder(TEST_PATH,  transform=val_tf)
    classes      = full_train.classes
    num_classes  = len(classes)

    train_idx, val_idx = train_test_split(
        np.arange(len(full_train)),
        test_size=0.2,
        stratify=full_train.targets,
        random_state=seed,
    )

    train_loader = DataLoader(
        Subset(full_train,   train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(full_val_ref, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    # -- Model / optimiser / scheduler ----------------------------------------
    model     = create_model(model_name, num_classes).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=1, eta_min=1e-6
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_val_loss    = float('inf')
    best_state       = None
    patience_counter = 0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    # -- Training loop --------------------------------------------------------
    for epoch in range(EPOCHS):
        torch.cuda.synchronize(DEVICE)
        torch.cuda.empty_cache()
        gc.collect()

        beta_kl    = BETA_KL_MAX   * min(1.0, epoch / 20.0)
        beta_proto = BETA_PROTO    * min(1.0, epoch / 10.0)

        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for batch_idx, (x, y) in enumerate(train_loader):
            try:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()

                if use_mixup:
                    x_mix, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)
                    # Pass labels for prototype loss (use y_a as dominant label)
                    logits, mu, logvar, ploss = model(x_mix,
                                                      labels=y_a if use_proto else None)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    logits, mu, logvar, ploss = model(x,
                                                      labels=y if use_proto else None)
                    loss = criterion(logits, y)

                if mu is not None:
                    kl   = -0.5 * torch.mean(
                        1 + logvar - mu.pow(2) - logvar.exp()
                    )
                    loss = loss + beta_kl * kl

                if use_proto and ploss is not None:
                    loss = loss + beta_proto * ploss

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                with torch.no_grad():
                    t_loss    += loss.item() * x.size(0)
                    t_correct += (logits.argmax(1) == y).sum().item()
                    t_total   += y.size(0)

            except RuntimeError as e:
                print(f"  ! Train batch {batch_idx} skipped: {e}")
                torch.cuda.empty_cache()
                continue

        scheduler.step()
        train_loss = t_loss    / max(t_total, 1)
        train_acc  = t_correct / max(t_total, 1)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits, _, _, _ = model(x)
                v_loss    += criterion(logits, y).item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total   += y.size(0)

        val_loss = v_loss    / max(v_total, 1)
        val_acc  = v_correct / max(v_total, 1)

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(round(train_loss, 6))
        history['train_acc'].append(round(train_acc,   6))
        history['val_loss'].append(round(val_loss,     6))
        history['val_acc'].append(round(val_acc,       6))

        print(f"  [{epoch+1:2d}/{EPOCHS}]  "
              f"Train {train_loss:.4f}/{train_acc:.4f}  "
              f"Val {val_loss:.4f}/{val_acc:.4f}  "
              f"beta_kl={beta_kl:.2e}  beta_proto={beta_proto:.2f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("  -> Early stopping triggered")
                break

    model.load_state_dict(best_state)

    # -- Save training history + curves (bold) -----------------------------
    pd.DataFrame(history).to_csv(paths["history_csv"], index=False)

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax1.plot(history['epoch'], history['train_loss'], 'b-',
             label='Train Loss', linewidth=2.5)
    ax1.plot(history['epoch'], history['val_loss'],   'c--',
             label='Val Loss',   linewidth=2.5)
    ax1.set_xlabel('Epoch', fontweight='bold', fontsize=13)
    ax1.set_ylabel('Loss', color='b', fontweight='bold', fontsize=13)
    ax1.tick_params(axis='y', labelcolor='b')
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
        lbl.set_fontweight('bold')
    ax2 = ax1.twinx()
    ax2.plot(history['epoch'], history['train_acc'], 'r-',
             label='Train Acc', alpha=0.85, linewidth=2.5)
    ax2.plot(history['epoch'], history['val_acc'],   'm--',
             label='Val Acc',   alpha=0.85, linewidth=2.5)
    ax2.set_ylabel('Accuracy', color='r', fontweight='bold', fontsize=13)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(0, 1.05)
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight('bold')
    lines  = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines],
               loc='center right', fontsize=10,
               prop={'weight': 'bold'})
    plt.title(f"{model_name.replace('_', ' ')}  —  seed {seed}",
              fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.25); plt.tight_layout()
    plt.savefig(paths["curves_png"], dpi=180, bbox_inches='tight')
    plt.close()

    # -- Test inference + per-seed GradCAM --------------------------------
    cam_base = paths["gradcam_base"]
    for cls in classes:
        (cam_base / cls).mkdir(parents=True, exist_ok=True)

    cam_engine = GradCAM(model, model.target_layer)
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    for i, (x, y) in enumerate(test_loader):
        try:
            x_dev = x.to(DEVICE)
            y_dev = y.to(DEVICE)

            with torch.set_grad_enabled(True):
                logits_eval, _, _, _ = model(x_dev)
                prob = F.softmax(logits_eval.detach(), dim=1)
                pred = logits_eval.argmax(dim=1).item()
                cam  = cam_engine.generate(x_dev, class_idx=y_dev.item())

            orig_img = denorm(x[0])
            cls_name = classes[y.item()]
            pred_name = classes[pred]
            correct   = (pred == y.item())

            # Save bold GradCAM figure (per seed per image)
            save_gradcam_figure(
                orig_img, cam, cls_name, pred_name, correct,
                model_name, seed, i, cam_base
            )

            y_true.append(y.item())
            y_pred.append(pred)
            y_prob.append(prob.cpu().numpy()[0])

        except Exception as e:
            print(f"  ! Test image {i} skipped: {e}")
            continue

    cam_engine.release()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # -- Metrics CSV ------------------------------------------------------
    acc, macro_f1, macro_prec, macro_rec, macro_auc = save_all_test_metrics(
        model_name, seed, classes, y_true, y_pred, y_prob,
        out_dir=dirs["metrics"]
    )

    # -- Confusion matrix (bold) ------------------------------------------
    cm    = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.to_csv(paths["cm_csv"])

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    sns.heatmap(
        cm_df, annot=True, fmt='d', cmap='Blues',
        linewidths=0.6, linecolor='gray', ax=ax,
        annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(
        f"Confusion Matrix\n{model_name.replace('_', ' ')}  —  Seed {seed}",
        fontweight='bold', fontsize=14
    )
    ax.set_ylabel("True Label",      fontweight='bold', fontsize=13)
    ax.set_xlabel("Predicted Label", fontweight='bold', fontsize=13)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(paths["cm_png"], dpi=180, bbox_inches='tight')
    plt.close()

    # -- ROC curves (bold) ------------------------------------------------
    roc_rows = []
    fig, ax  = plt.subplots(figsize=(9, 6.5))
    for j, cls in enumerate(classes):
        fpr, tpr, thresholds = roc_curve((y_true == j).astype(int), y_prob[:, j])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})',
                linewidth=2.5)
        for f, t, th in zip(fpr, tpr, thresholds):
            roc_rows.append({
                'model': model_name, 'seed': seed, 'class': cls,
                'fpr': round(float(f), 6), 'tpr': round(float(t), 6),
                'threshold': round(float(th), 6), 'auc': round(roc_auc, 6),
            })
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5)
    legend = ax.legend(fontsize=10, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3)
    ax.set_title(f"ROC Curves\n{model_name.replace('_', ' ')}  —  Seed {seed}",
                 fontweight='bold', fontsize=14)
    ax.set_xlabel("False Positive Rate", fontweight='bold', fontsize=13)
    ax.set_ylabel("True Positive Rate",  fontweight='bold', fontsize=13)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(paths["roc_png"], dpi=180, bbox_inches='tight')
    plt.close()
    pd.DataFrame(roc_rows).to_csv(paths["roc_data"], index=False)

    result = {
        "seed":            seed,
        "model":           model_name,
        "acc":             round(acc,        4),
        "f1_macro":        round(macro_f1,   4),
        "precision_macro": round(macro_prec, 4),
        "recall_macro":    round(macro_rec,  4),
        "auc_macro":       round(macro_auc,  4),
    }
    print(f"  [ok]  Acc={acc:.4f}  F1={macro_f1:.4f}  AUC={macro_auc:.4f}")
    return result


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2)
        / (na + nb - 2 + 1e-12)
    )
    return float((a.mean() - b.mean()) / (pooled + 1e-12))


def effect_label(d: float) -> str:
    a = abs(d)
    if a >= 0.8:  return "large"
    if a >= 0.5:  return "medium"
    if a >= 0.2:  return "small"
    return "negligible"


def load_predictions_from_cm(model_name: str, seed: int):
    cm_path = SAVE_DIR / model_name / "metrics" / f"cm_seed{seed}.csv"
    if not cm_path.exists():
        raise FileNotFoundError(f"Missing: {cm_path}")
    cm_df   = pd.read_csv(cm_path, index_col=0)
    classes = list(cm_df.index)
    cm      = cm_df.values.astype(int)
    y_true_list, y_pred_list = [], []
    for ti in range(len(classes)):
        for pi in range(len(classes)):
            count = cm[ti, pi]
            y_true_list.extend([ti] * count)
            y_pred_list.extend([pi] * count)
    return np.array(y_true_list), np.array(y_pred_list), classes


def mcnemar_test(y_true, pred_a, pred_b):
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)
    b = int(np.sum( correct_a & ~correct_b))
    c = int(np.sum(~correct_a &  correct_b))
    if (b + c) == 0:
        return np.nan, np.nan, b, c
    chi2_stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p_value   = 1.0 - chi2.cdf(chi2_stat, df=1)
    return chi2_stat, p_value, b, c


# -----------------------------------------------------------------------------
# Wilcoxon signed-rank (n=6)
# -----------------------------------------------------------------------------
def run_wilcoxon(df: pd.DataFrame):
    print("\n=== Wilcoxon Signed-Rank Test (n=6 seeds) ===")
    rows = []
    for metric in METRICS:
        print(f"\n  Metric: {metric}")
        for m1, m2 in itertools.combinations(MODELS, 2):
            s1 = df[df["model"] == m1][metric].values
            s2 = df[df["model"] == m2][metric].values
            if len(s1) < 2:
                continue
            try:
                w_stat, p = stats.wilcoxon(s1, s2, zero_method='wilcox',
                                            correction=False)
            except ValueError:
                w_stat, p = np.nan, np.nan

            d     = cohens_d(s1, s2)
            delta = s1.mean() - s2.mean()
            eff   = effect_label(d)
            sig   = (not np.isnan(p)) and (p < ALPHA)

            print(f"    {m1:35s} vs {m2:30s} | "
                  f"d={delta:+.4f}  W={w_stat}  p={p:.4f}  "
                  f"d={d:+.3f}({eff})  {'* sig' if sig else 'n.s.'}")

            rows.append({
                "metric":          metric,
                "model_A":         m1,
                "model_B":         m2,
                "mean_A":          round(s1.mean(), 4),
                "mean_B":          round(s2.mean(), 4),
                "delta_A_minus_B": round(delta,     6),
                "wilcoxon_W":      round(w_stat, 2) if not np.isnan(w_stat) else np.nan,
                "p_value":         round(p, 6)       if not np.isnan(p)      else np.nan,
                "cohens_d":        round(d, 4),
                "effect_size":     eff,
                "significant_005": sig,
                "n_seeds":         len(s1),
            })

    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "wilcoxon_n6.csv", index=False)
    print(f"\n  [ok]  Saved wilcoxon_n6.csv")
    return out


# -----------------------------------------------------------------------------
# McNemar pooled
# -----------------------------------------------------------------------------
def run_mcnemar_pooled(all_seeds):
    print("\n=== McNemar's Test — Pooled Across All Seeds (PRIMARY STAT) ===")
    pair_counts = {}

    for seed in all_seeds:
        preds = {}
        y_ref = None
        for m in MODELS:
            try:
                y_true, y_pred, _ = load_predictions_from_cm(m, seed)
                preds[m] = y_pred
                if y_ref is None:
                    y_ref = y_true
            except FileNotFoundError as e:
                print(f"  ! {e}")
                continue

        for m1, m2 in itertools.combinations(list(preds.keys()), 2):
            _, _, b, c = mcnemar_test(y_ref, preds[m1], preds[m2])
            key = (m1, m2)
            if key not in pair_counts:
                pair_counts[key] = [0, 0]
            pair_counts[key][0] += b
            pair_counts[key][1] += c

    rows = []
    print(f"\n  {'Model A':35s} vs {'Model B':33s} | "
          f"{'b':>6} {'c':>6}  {'chi2':>8}  {'p':>8}  result  winner")
    print("  " + "-" * 110)

    for (m1, m2), (b_total, c_total) in pair_counts.items():
        if (b_total + c_total) == 0:
            chi2_stat, p = np.nan, np.nan
        else:
            chi2_stat = (abs(b_total - c_total) - 1.0) ** 2 / (b_total + c_total)
            p         = 1.0 - chi2.cdf(chi2_stat, df=1)

        sig    = (not np.isnan(p)) and (p < ALPHA)
        winner = m1 if b_total > c_total else (m2 if c_total > b_total else "tie")

        print(f"  {m1:35s} vs {m2:33s} | "
              f"{b_total:6d} {c_total:6d}  "
              f"{chi2_stat:8.3f}  {p:8.4f}  "
              f"{'SIGNIFICANT' if sig else 'n.s.':12s}  {winner}")

        rows.append({
            "model_A":         m1,
            "model_B":         m2,
            "pooled_b_A_wins": b_total,
            "pooled_c_B_wins": c_total,
            "chi2":            round(chi2_stat, 4) if not np.isnan(chi2_stat) else np.nan,
            "p_value":         round(p, 6)         if not np.isnan(p)         else np.nan,
            "significant_005": sig,
            "winner":          winner,
            "n_seeds_pooled":  len(all_seeds),
        })

    out = pd.DataFrame(rows)
    out.to_csv(SAVE_DIR / "mcnemar_pooled.csv", index=False)
    print(f"\n  [ok]  Saved mcnemar_pooled.csv")
    return out


# -----------------------------------------------------------------------------
# Superiority table
# -----------------------------------------------------------------------------
def build_superiority_table(df_results, wilcoxon_df, mcnemar_df):
    proposed = "MCFA_Net_EfficientNet_B3"
    baseline = "EfficientNet_B3"
    rows     = []

    print("\n\n" + "="*75)
    print("  SUPERIORITY TABLE: MCFA_Net_EfficientNet_B3  vs  EfficientNet_B3")
    print("="*75)

    for metric in METRICS:
        s_prop = df_results[df_results["model"] == proposed][metric].values
        s_base = df_results[df_results["model"] == baseline][metric].values

        mean_p, std_p = s_prop.mean(), s_prop.std(ddof=1)
        mean_b, std_b = s_base.mean(), s_base.std(ddof=1)
        delta         = mean_p - mean_b
        d             = cohens_d(s_prop, s_base)

        wilc = wilcoxon_df[
            (wilcoxon_df["metric"] == metric) &
            (
                ((wilcoxon_df["model_A"] == proposed) & (wilcoxon_df["model_B"] == baseline)) |
                ((wilcoxon_df["model_A"] == baseline) & (wilcoxon_df["model_B"] == proposed))
            )
        ]
        p_wilcoxon = wilc.iloc[0]["p_value"] if not wilc.empty else np.nan

        mc = mcnemar_df[
            ((mcnemar_df["model_A"] == proposed) & (mcnemar_df["model_B"] == baseline)) |
            ((mcnemar_df["model_A"] == baseline) & (mcnemar_df["model_B"] == proposed))
        ]
        p_mcnemar = mc.iloc[0]["p_value"] if not mc.empty else np.nan
        b_val     = mc.iloc[0]["pooled_b_A_wins"] if not mc.empty else np.nan
        c_val     = mc.iloc[0]["pooled_c_B_wins"] if not mc.empty else np.nan
        if not mc.empty and mc.iloc[0]["model_A"] == baseline:
            b_val, c_val = c_val, b_val

        sv_wins = (delta > 0) and (
            (not np.isnan(p_mcnemar)  and p_mcnemar  < ALPHA) or
            (not np.isnan(p_wilcoxon) and p_wilcoxon < ALPHA)
        )
        rows.append({
            "Metric":                       METRIC_LABELS[metric],
            "MCFA-Net Mean+-Std":           f"{mean_p:.4f} +- {std_p:.4f}",
            "EfficientNet-B3 Mean+-Std":    f"{mean_b:.4f} +- {std_b:.4f}",
            "Delta (MCFA-Net minus B3)":    f"{delta:+.4f}",
            "Cohens d":                     f"{d:.4f}",
            "Effect size":                  effect_label(abs(d)),
            "Wilcoxon p (n=6)":             f"{p_wilcoxon:.4f}" if not np.isnan(p_wilcoxon) else "n/a",
            "McNemar p (pooled)":           f"{p_mcnemar:.4f}"  if not np.isnan(p_mcnemar)  else "n/a",
            "McNemar b (MCFA-Net correct)": int(b_val) if not np.isnan(b_val) else "?",
            "McNemar c (B3 correct)":       int(c_val) if not np.isnan(c_val) else "?",
            "MCFA-Net better":              "YES *" if sv_wins else ("numerically" if delta > 0 else "NO"),
        })

    sup_df = pd.DataFrame(rows)
    sup_df.to_csv(SAVE_DIR / "superiority_table.csv", index=False)
    print(sup_df.to_string(index=False))
    print(f"\n  [ok]  Saved superiority_table.csv")
    return sup_df


# -----------------------------------------------------------------------------
# Ablation superiority
# -----------------------------------------------------------------------------
def build_ablation_table(df_results, wilcoxon_df, mcnemar_df):
    baseline    = "EfficientNet_B3"
    comparators = [m for m in MODELS if m != baseline]
    rows        = []

    for proposed in comparators:
        for metric in METRICS:
            s_prop = df_results[df_results["model"] == proposed][metric].values
            s_base = df_results[df_results["model"] == baseline][metric].values
            delta  = s_prop.mean() - s_base.mean()
            d      = cohens_d(s_prop, s_base)

            wilc = wilcoxon_df[
                (wilcoxon_df["metric"] == metric) &
                (
                    ((wilcoxon_df["model_A"] == proposed) & (wilcoxon_df["model_B"] == baseline)) |
                    ((wilcoxon_df["model_A"] == baseline) & (wilcoxon_df["model_B"] == proposed))
                )
            ]
            p_wilcoxon = wilc.iloc[0]["p_value"] if not wilc.empty else np.nan

            mc = mcnemar_df[
                ((mcnemar_df["model_A"] == proposed) & (mcnemar_df["model_B"] == baseline)) |
                ((mcnemar_df["model_A"] == baseline) & (mcnemar_df["model_B"] == proposed))
            ]
            p_mcnemar = mc.iloc[0]["p_value"] if not mc.empty else np.nan

            rows.append({
                "Model":              proposed,
                "Metric":             METRIC_LABELS[metric],
                "Delta vs B3":        f"{delta:+.4f}",
                "Cohens d":           f"{d:.4f}",
                "Effect":             effect_label(abs(d)),
                "Wilcoxon p (n=6)":   f"{p_wilcoxon:.4f}" if not np.isnan(p_wilcoxon) else "n/a",
                "McNemar p (pooled)": f"{p_mcnemar:.4f}"  if not np.isnan(p_mcnemar)  else "n/a",
                "Direction":          "better" if delta > 0 else ("worse" if delta < 0 else "equal"),
            })

    abl_df = pd.DataFrame(rows)
    abl_df.to_csv(SAVE_DIR / "ablation_superiority.csv", index=False)
    print(f"\n  [ok]  Saved ablation_superiority.csv")
    return abl_df


# -----------------------------------------------------------------------------
# Publication-quality comparison plots (bold)
# -----------------------------------------------------------------------------
def _bold_axes(ax):
    """Make all tick labels bold."""
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')


SHORT = {
    "EfficientNet_B3":            "EffNet-B3",
    "MCFA_no_DDAF":               "No-DDAF",
    "MCFA_no_HPCB":               "No-HPCB",
    "MCFA_no_AMPP":               "No-AMPP",
    "MCFA_Net_EfficientNet_B3":   "MCFA-Net\n(Proposed)",
}


def save_comparison_plots(df: pd.DataFrame, agg: pd.DataFrame, all_seeds):

    palette = sns.color_palette("Set2", len(MODELS))
    x       = np.arange(len(MODELS))
    labels  = [SHORT.get(m, m) for m in MODELS]

    # -- Box plots --------------------------------------------------------
    fig, axes = plt.subplots(1, len(METRICS), figsize=(24, 5.5), sharey=False)
    for ax, metric in zip(axes, METRICS):
        sns.boxplot(x="model", y=metric, data=df, order=MODELS,
                    palette=palette, width=0.5, ax=ax, linewidth=2)
        sns.stripplot(x="model", y=metric, data=df, order=MODELS,
                      color="k", size=6, jitter=0.15, ax=ax)
        ax.set_title(METRIC_LABELS[metric], fontweight='bold', fontsize=13)
        ax.set_xlabel("")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9,
                           fontweight='bold')
        ax.set_ylabel(METRIC_LABELS[metric], fontweight='bold', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        _bold_axes(ax)
    plt.suptitle(
        f"MCFA-Net  —  Performance Distribution  (n={len(all_seeds)} seeds)",
        fontweight='bold', fontsize=15
    )
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "boxplot_all_metrics.png", dpi=200, bbox_inches='tight')
    plt.close()

    # -- Bar plots --------------------------------------------------------
    for metric in METRICS:
        col_mean = f"{metric}_mean"
        col_std  = f"{metric}_std"
        if col_mean not in agg.columns:
            continue
        means = agg[col_mean].values
        stds  = agg[col_std].values

        fig, ax = plt.subplots(figsize=(13, 6.5))
        bars = ax.bar(x, means, yerr=stds, capsize=7,
                      color=palette, edgecolor='k', linewidth=1.2)
        # Highlight proposed in distinct colour
        bars[-1].set_edgecolor('navy')
        bars[-1].set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right",
                           fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Mean {METRIC_LABELS[metric]} ± Std",
                      fontweight='bold', fontsize=13)
        ax.set_title(
            f"{METRIC_LABELS[metric]}  Comparison  (n={len(all_seeds)} seeds)",
            fontweight='bold', fontsize=14
        )
        ax.set_ylim(0, 1.10)
        ax.grid(axis='y', alpha=0.3)
        _bold_axes(ax)

        for bar, m, s in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 0.006,
                f"{m:.4f}",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold'
            )
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"barplot_{metric}.png", dpi=200, bbox_inches='tight')
        plt.close()

    # -- McNemar p-value heatmap (bold) ----------------------------------
    mc_df = pd.read_csv(SAVE_DIR / "mcnemar_pooled.csv")
    p_mat = pd.DataFrame(
        np.ones((len(MODELS), len(MODELS))), index=MODELS, columns=MODELS
    )
    for _, row in mc_df.iterrows():
        p = row["p_value"]
        if np.isnan(p):
            continue
        p_mat.loc[row["model_A"], row["model_B"]] = p
        p_mat.loc[row["model_B"], row["model_A"]] = p

    short_rename = {m: SHORT.get(m, m) for m in MODELS}
    p_mat_plot = p_mat.rename(index=short_rename, columns=short_rename)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        p_mat_plot.astype(float), annot=True, fmt=".4f",
        cmap="RdYlGn_r", vmin=0, vmax=0.1,
        mask=np.eye(len(MODELS), dtype=bool), ax=ax,
        linewidths=0.5, linecolor="white",
        annot_kws={"size": 12, "weight": "bold"},
        cbar_kws={"label": f"McNemar p-value (pooled, n={len(all_seeds)} seeds)"}
    )
    ax.set_title("Pairwise McNemar p-values", fontweight='bold', fontsize=14)
    plt.xticks(rotation=30, ha="right", fontsize=11, fontweight='bold')
    plt.yticks(rotation=0,  fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "mcnemar_pvalue_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    # -- Cohen's d heatmap (bold) -----------------------------------------
    wilc_df = pd.read_csv(SAVE_DIR / "wilcoxon_n6.csv")
    d_mat   = pd.DataFrame(index=MODELS, columns=METRICS, dtype=float)
    for _, row in wilc_df.iterrows():
        d_mat.loc[row["model_A"], row["metric"]] =  row["cohens_d"]
        d_mat.loc[row["model_B"], row["metric"]] = -row["cohens_d"]
    d_mat = d_mat.rename(columns=METRIC_LABELS, index=short_rename)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(
        d_mat.astype(float), annot=True, fmt=".3f",
        cmap="RdBu", center=0, ax=ax,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        cbar_kws={"label": "Cohen's d"}
    )
    ax.set_title(
        f"Cohen's d Effect Sizes  (n={len(all_seeds)} seeds)",
        fontweight='bold', fontsize=14
    )
    plt.xticks(rotation=20, ha="right", fontsize=11, fontweight='bold')
    plt.yticks(rotation=0,  fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "cohens_d_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()

    print("  [ok]  All comparison plots saved.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    print("\n" + "="*70)
    print("  MCFA-Net End-to-End Training + Statistical Analysis")
    print(f"  Models : {len(MODELS)}  |  Seeds : {SEEDS}  (n={len(SEEDS)})")
    print(f"  Device : {DEVICE}")
    print("="*70)
    print("""
  NOVELTY SUMMARY
  ---------------
  Component 1  DDAF  — Dual-Domain Attention Fusion
               Learnable per-sample gate mixing spatial & frequency (Wavelet+FFT)
               feature maps. Gate is input-content-aware.

  Component 2  AMPP  — Adaptive Multi-scale Pyramid Pooling
               Pools at 4 spatial scales with self-attention scale weighting
               that is input-dependent (replaces fixed GAP).

  Component 3  HPCB  — Hierarchical Prototype Contrastive Bottleneck
               Per-class learnable prototypes + prototype-conditioned
               variational bottleneck + contrastive loss = structured latent space.
  """)

    all_results = []

    # -- 1. Train all models × all seeds ----------------------------------
    for model_name in MODELS:
        for seed in SEEDS:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            res = run_one_seed(seed, model_name)
            all_results.append(res)

            # Append to CSV incrementally (crash recovery)
            df_inc = pd.DataFrame(all_results)
            df_inc = df_inc.drop_duplicates(subset=["model", "seed"])
            df_inc.to_csv(SAVE_DIR / "all_results.csv", index=False)

    # -- 2. Aggregate ------------------------------------------------------
    df = pd.DataFrame(all_results)
    df = df.drop_duplicates(subset=["model", "seed"]).reset_index(drop=True)
    df.to_csv(SAVE_DIR / "all_results.csv", index=False)

    agg_cols = {
        "acc":             ["mean", "std", "min", "max"],
        "f1_macro":        ["mean", "std"],
        "precision_macro": ["mean", "std"],
        "recall_macro":    ["mean", "std"],
        "auc_macro":       ["mean", "std"],
    }
    agg = df.groupby("model").agg(agg_cols).round(4)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reindex(MODELS)
    agg.to_csv(SAVE_DIR / "summary_aggregated.csv")

    print("\n\n=== Aggregated Summary ===")
    print(agg.to_string())

    agg.loc[[m for m in MODELS if m in agg.index]].to_csv(
        SAVE_DIR / "ablation_summary.csv"
    )

    # -- 3. Wilcoxon -------------------------------------------------------
    wilcoxon_df = run_wilcoxon(df)

    # -- 4. McNemar pooled -------------------------------------------------
    mcnemar_df = run_mcnemar_pooled(SEEDS)

    # -- 5. Superiority + ablation tables ---------------------------------
    sup_df = build_superiority_table(df, wilcoxon_df, mcnemar_df)
    abl_df = build_ablation_table(df, wilcoxon_df, mcnemar_df)

    # -- 6. Comparison plots -----------------------------------------------
    save_comparison_plots(df, agg, SEEDS)

    # -- 7. Final summary --------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  All outputs saved to: {SAVE_DIR.resolve()}")
    print(f"{'='*70}")
    print(f"""
  Per-model subfolders (5 models):
    EfficientNet_B3/           metrics/  curves/  gradcam/<class>/img_NNNNN_seed<s>.png
    MCFA_no_DDAF/              metrics/  curves/  gradcam/<class>/img_NNNNN_seed<s>.png
    MCFA_no_HPCB/              metrics/  curves/  gradcam/<class>/img_NNNNN_seed<s>.png
    MCFA_no_AMPP/              metrics/  curves/  gradcam/<class>/img_NNNNN_seed<s>.png
    MCFA_Net_EfficientNet_B3/  metrics/  curves/  gradcam/<class>/img_NNNNN_seed<s>.png

  Global outputs:
    all_results.csv            — raw seed-level results
    summary_aggregated.csv     — mean/std/min/max per model
    ablation_summary.csv       — ablation-focused view
    wilcoxon_n6.csv            — Wilcoxon signed-rank (n=6)
    mcnemar_pooled.csv         — McNemar pooled (PRIMARY STAT)
    superiority_table.csv      — paper-ready MCFA-Net vs B3
    ablation_superiority.csv   — all ablations vs B3
    boxplot_all_metrics.png    — distribution across seeds
    barplot_{{metric}}.png x5   — mean±std per metric
    mcnemar_pvalue_heatmap.png — pairwise p-value grid
    cohens_d_heatmap.png       — effect sizes
    """)
