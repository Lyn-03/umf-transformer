"""
UFM-Transformer: Complete Evaluation and Baseline Comparison Module.

This module provides comprehensive evaluation capabilities for the
Uncertainty-aware Fusion Module (UFM) Transformer model, including:

    1. Biometric evaluation metrics (EER, TAR@FAR, AUC, FNMR/FMR, DET)
    2. Baseline model architectures for comparison:
       - ConcatenationFusion: Simple feature concatenation with MLP
       - ScoreSumFusion: Independent encoders with weighted score fusion
       - DenseNetBiModal: DenseNet processing concatenated modalities
       - TransformerFusionNoUncertainty: UFM without uncertainty head
       - MDRLFusion: Multi-level hierarchical representation learning
    3. Evaluation protocol: standard, robustness, uncertainty calibration
    4. Visualization functions (7 plot types) exported as publication-quality PDFs
    5. LaTeX table export for academic papers

All models share a unified interface:
    forward(face_img, fp_img, face_quality, fp_quality, missing_mask) -> (score, None|unc1, None|unc2)

Example usage:
    $ python evaluate.py --model_path ./checkpoints/best_model.pth \\
        --dataset_path /path/to/data --output_dir ./results

Author: Biometrics Pipeline Team
Python Version: >=3.10
PyTorch Version: >=2.0
n"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as TF

# Matplotlib configuration for publication-quality figures
matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.figsize"] = (8, 6)

# Ensure project root is on path for local imports
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import get_dataloaders

# UFMTransformer is imported from models.py; other baselines are self-contained.
# The TransformerFusionNoUncertainty baseline below re-implements needed components
# to avoid dependency on model-specific internal classes.
try:
    from models import UFMTransformer
except ImportError as _import_err:
    warnings.warn(f"Could not import UFMTransformer from models.py: {_import_err}")
    UFMTransformer = nn.Module  # type: ignore[misc,assignment]

# Color constants for plotting
C_GEN = "#2E86AB"
C_IMP = "#A23B72"
C_ACC = "#F18F01"
C_OK = "#27AE60"
C_ERR = "#E74C3C"


# =============================================================================
# 1. EVALUATION METRICS
# =============================================================================

def compute_eer(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and corresponding threshold.

    The EER is the point where False Acceptance Rate (FAR) equals
    False Rejection Rate (FRR = 1 - TAR). This is a standard
    single-number summary of biometric system performance.

    Args:
        scores_genuine: Similarity scores for genuine pairs (N_genuine,).
        scores_impostor: Similarity scores for impostor pairs (N_impostor,).

    Returns:
        Tuple of (eer, threshold) where eer is in percent (0-100).

    Raises:
        ValueError: If either score array is empty.
    """
    if len(scores_genuine) == 0 or len(scores_impostor) == 0:
        raise ValueError("Both genuine and impostor score arrays must be non-empty")

    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([
        np.ones(len(scores_genuine), dtype=int),
        np.zeros(len(scores_impostor), dtype=int),
    ])
    sorted_idx = np.argsort(all_scores)
    sorted_scores = all_scores[sorted_idx]
    sorted_labels = all_labels[sorted_idx]

    cumsum_gen = np.cumsum(sorted_labels)
    total_gen = np.sum(sorted_labels)
    total_imp = len(sorted_labels) - total_gen

    fnmr = cumsum_gen / total_gen if total_gen > 0 else np.zeros_like(cumsum_gen)
    fp_count = total_imp - (np.arange(1, len(sorted_labels) + 1) - cumsum_gen)
    fmr = fp_count / total_imp if total_imp > 0 else np.zeros_like(fp_count)

    diffs = np.abs(fmr - fnmr)
    eer_idx = int(np.argmin(diffs))
    eer = float((fmr[eer_idx] + fnmr[eer_idx]) / 2.0 * 100.0)
    if eer_idx + 1 < len(sorted_scores):
        threshold = float((sorted_scores[eer_idx] + sorted_scores[eer_idx + 1]) / 2.0)
    else:
        threshold = float(sorted_scores[eer_idx])
    return eer, threshold


def compute_tar_at_far(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
    far_targets: List[float] = [0.001, 0.01],
) -> Dict[float, float]:
    """Compute True Acceptance Rate (TAR) at specified False Acceptance Rates (FAR).

    TAR@FAR=0.1% measures the proportion of genuine pairs correctly accepted
    when the system is configured to accept at most 0.1% of impostors.

    Args:
        scores_genuine: Genuine pair similarity scores (N_genuine,).
        scores_impostor: Impostor pair similarity scores (N_impostor,).
        far_targets: FAR values to compute TAR at. Defaults to [0.001, 0.01].

    Returns:
        Dictionary mapping each FAR target to TAR percentage (0-100).

    Raises:
        ValueError: If either score array is empty.
    """
    if len(scores_genuine) == 0 or len(scores_impostor) == 0:
        raise ValueError("Both genuine and impostor score arrays must be non-empty")

    results = {}
    sorted_imp = np.sort(scores_impostor)
    n_imp = len(sorted_imp)
    n_gen = len(scores_genuine)

    for far_target in far_targets:
        max_accepted = max(0, min(int(np.floor(far_target * n_imp)), n_imp - 1))
        if max_accepted < n_imp:
            idx = max(0, min(n_imp - max_accepted - 1, n_imp - 1))
            threshold = sorted_imp[idx]
        else:
            threshold = sorted_imp[-1] + 1e-6
        tar = float(np.sum(scores_genuine >= threshold) / n_gen * 100.0)
        results[far_target] = tar
    return results


def compute_auc(scores_genuine: np.ndarray, scores_impostor: np.ndarray) -> float:
    """Compute Area Under the ROC Curve (AUC-ROC) as percentage.

    Args:
        scores_genuine: Genuine pair similarity scores.
        scores_impostor: Impostor pair similarity scores.

    Returns:
        AUC-ROC as percentage (0-100%).

    Raises:
        ValueError: If either score array is empty.
    """
    if len(scores_genuine) == 0 or len(scores_impostor) == 0:
        raise ValueError("Both genuine and impostor score arrays must be non-empty")
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([
        np.ones(len(scores_genuine), dtype=int),
        np.zeros(len(scores_impostor), dtype=int),
    ])
    try:
        return float(roc_auc_score(all_labels, all_scores) * 100.0)
    except ValueError:
        return 50.0


def compute_fnmr_fmr(
    scores_genuine: np.ndarray, scores_impostor: np.ndarray, threshold: float
) -> Tuple[float, float]:
    """Compute FNMR and FMR at a given operating threshold.

    Args:
        scores_genuine: Genuine pair similarity scores.
        scores_impostor: Impostor pair similarity scores.
        threshold: Operating point score threshold.

    Returns:
        Tuple of (fnmr, fmr) as percentages.

    Raises:
        ValueError: If either score array is empty.
    """
    if len(scores_genuine) == 0 or len(scores_impostor) == 0:
        raise ValueError("Both genuine and impostor score arrays must be non-empty")
    fnmr = float(np.sum(scores_genuine < threshold) / len(scores_genuine) * 100.0)
    fmr = float(np.sum(scores_impostor >= threshold) / len(scores_impostor) * 100.0)
    return fnmr, fmr


def compute_det_curve(
    scores_genuine: np.ndarray, scores_impostor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Detection Error Tradeoff (DET) curve coordinates.

    Args:
        scores_genuine: Genuine pair similarity scores.
        scores_impostor: Impostor pair similarity scores.

    Returns:
        Tuple of (fmr_rates, fnmr_rates, thresholds):
            - fmr_rates: FMR values (0-100%).
            - fnmr_rates: FNMR values (0-100%).
            - thresholds: Corresponding thresholds.

    Raises:
        ValueError: If either score array is empty.
    """
    if len(scores_genuine) == 0 or len(scores_impostor) == 0:
        raise ValueError("Both genuine and impostor score arrays must be non-empty")
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([
        np.ones(len(scores_genuine), dtype=int),
        np.zeros(len(scores_impostor), dtype=int),
    ])
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores, drop_intermediate=True)
    fmr_rates = fpr * 100.0
    fnmr_rates = (1.0 - tpr) * 100.0
    return fmr_rates, fnmr_rates, thresholds


def compute_calibration_error(
    scores: np.ndarray, labels: np.ndarray, n_bins: int = 10
) -> Tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Bins predicted scores into n_bins equal-width bins and computes the
    average absolute difference between accuracy and confidence per bin.

    Args:
        scores: Predicted confidence scores (N,) in [0, 1].
        labels: True labels (N,) where 1 = genuine, 0 = impostor.
        n_bins: Number of calibration bins. Defaults to 10.

    Returns:
        Tuple of (ece, mce) as percentages (0-100%).

    Raises:
        ValueError: If scores and labels are empty or have different lengths.
    """
    if len(scores) == 0 or len(labels) == 0:
        raise ValueError("Scores and labels must be non-empty")
    if len(scores) != len(labels):
        raise ValueError(f"Scores and labels must have same length, got {len(scores)} vs {len(labels)}")

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    total = len(scores)

    for i in range(n_bins):
        lower, upper = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = (scores >= lower) & (scores <= upper)
        else:
            in_bin = (scores >= lower) & (scores < upper)
        bin_size = int(np.sum(in_bin))
        if bin_size == 0:
            continue
        bin_conf = float(np.mean(scores[in_bin]))
        bin_acc = float(np.mean(labels[in_bin]))
        error = abs(bin_acc - bin_conf)
        ece += (bin_size / total) * error * 100.0
        mce = max(mce, error * 100.0)

    return float(ece), float(mce)


def compute_all_metrics(
    scores_genuine: np.ndarray, scores_impostor: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics in a single call.

    Args:
        scores_genuine: Genuine pair similarity scores.
        scores_impostor: Impostor pair similarity scores.

    Returns:
        Dictionary with all metrics: eer, eer_threshold, tar_at_0_001,
        tar_at_0_01, auc, fnmr_at_eer, fmr_at_eer, ece, mce.
    """
    eer, eer_thr = compute_eer(scores_genuine, scores_impostor)
    tar_dict = compute_tar_at_far(scores_genuine, scores_impostor, [0.001, 0.01])
    auc_val = compute_auc(scores_genuine, scores_impostor)
    fnmr, fmr = compute_fnmr_fmr(scores_genuine, scores_impostor, eer_thr)

    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([
        np.ones(len(scores_genuine), dtype=int),
        np.zeros(len(scores_impostor), dtype=int),
    ])
    ece, mce = compute_calibration_error(all_scores, all_labels, n_bins=10)

    return {
        "eer": eer,
        "eer_threshold": eer_thr,
        "tar_at_0_001": tar_dict[0.001],
        "tar_at_0_01": tar_dict[0.01],
        "auc": auc_val,
        "fnmr_at_eer": fnmr,
        "fmr_at_eer": fmr,
        "ece": ece,
        "mce": mce,
    }



# =============================================================================
# 2. BASELINE MODELS
# =============================================================================
# All baselines share the unified interface:
#   forward(face_img, fp_img, face_quality, fp_quality, missing_mask) -> (score, None, None)


class ConcatenationFusion(nn.Module):
    """Simple concatenation fusion baseline for biometric verification.

    Extracts features independently from face and fingerprint images using
    lightweight CNN encoders, concatenates the features, and passes them
    through a multi-layer perceptron (MLP) to produce a verification score.

    This is the simplest possible fusion strategy and serves as a lower
    bound for comparison.

    Args:
        feature_dim: Dimensionality of each modality's feature vector. Defaults to 256.
        dropout: Dropout probability. Defaults to 0.1.

    Input / Output:
        Same unified interface as UFMTransformer (see module docstring).
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        def _make_encoder(in_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.face_encoder = _make_encoder(3)
        self.fp_encoder = _make_encoder(3)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        face_img: Tensor,
        fp_img: Tensor,
        face_quality: Union[Tensor, float],
        fp_quality: Union[Tensor, float],
        missing_mask: Tensor,
    ) -> Tuple[Tensor, None, None]:
        """Forward pass of concatenation fusion baseline.

        Args:
            face_img: Face images (N, 3, H, W).
            fp_img: Fingerprint images (N, 3, H, W).
            face_quality: Face quality scores (N,) or (N, 1).
            fp_quality: Fingerprint quality scores (N,) or (N, 1).
            missing_mask: Missing modality mask (N, 2).

        Returns:
            Tuple of (score, None, None).
        """
        face_feat = self.face_encoder(face_img)
        fp_feat = self.fp_encoder(fp_img)
        fused = torch.cat([face_feat, fp_feat], dim=1)
        score = self.classifier(fused).squeeze(-1)
        return score, None, None


class ScoreSumFusion(nn.Module):
    """Score-level weighted sum fusion baseline.

    Uses independent encoders for each modality, computes similarity
    per modality, and fuses using a quality-weighted sum.

    Args:
        feature_dim: Dimensionality of each modality's feature vector. Defaults to 256.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(self, feature_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim

        def _make_encoder(in_ch: int) -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.face_encoder = _make_encoder(3)
        self.fp_encoder = _make_encoder(3)

        self.quality_gate = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Softmax(dim=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        face_img: Tensor,
        fp_img: Tensor,
        face_quality: Union[Tensor, float],
        fp_quality: Union[Tensor, float],
        missing_mask: Tensor,
    ) -> Tuple[Tensor, None, None]:
        """Forward pass of score-sum fusion baseline.

        Args:
            face_img: Face images (N, 3, H, W).
            fp_img: Fingerprint images (N, 3, H, W).
            face_quality: Face quality scores (N,) or (N, 1).
            fp_quality: Fingerprint quality scores (N,) or (N, 1).
            missing_mask: Missing modality mask (N, 2).

        Returns:
            Tuple of (score, None, None).
        """
        N = face_img.shape[0]
        if isinstance(face_quality, (int, float)):
            face_quality = torch.full((N,), face_quality, device=face_img.device, dtype=torch.float32)
        if isinstance(fp_quality, (int, float)):
            fp_quality = torch.full((N,), fp_quality, device=face_img.device, dtype=torch.float32)
        if face_quality.dim() == 0:
            face_quality = face_quality.unsqueeze(0).expand(N)
        if fp_quality.dim() == 0:
            fp_quality = fp_quality.unsqueeze(0).expand(N)
        if face_quality.dim() == 1:
            face_quality = face_quality.unsqueeze(1)
        if fp_quality.dim() == 1:
            fp_quality = fp_quality.unsqueeze(1)

        face_feat = F.normalize(self.face_encoder(face_img), p=2, dim=1)
        fp_feat = F.normalize(self.fp_encoder(fp_img), p=2, dim=1)

        face_score = (face_feat * face_feat).sum(dim=1) * 0.5 + 0.5
        fp_score = (fp_feat * fp_feat).sum(dim=1) * 0.5 + 0.5

        quality_input = torch.cat([face_quality, fp_quality], dim=1)
        weights = self.quality_gate(quality_input)

        face_missing = missing_mask[:, 0]
        fp_missing = missing_mask[:, 1]
        weights = weights.clone()
        weights[face_missing.bool(), 0] = 0.0
        weights[fp_missing.bool(), 1] = 0.0
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        weights = weights / weight_sum

        score = weights[:, 0] * face_score + weights[:, 1] * fp_score
        return score, None, None


# ---- DenseNet components ----

class _DenseLayer(nn.Module):
    """Single dense layer for DenseNet.

    Args:
        in_channels: Number of input channels.
        growth_rate: Number of feature maps to produce. Defaults to 32.
        bn_size: Bottleneck size. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self, in_channels: int, growth_rate: int = 32, bn_size: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        mid_channels = bn_size * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, growth_rate, 3, padding=1, bias=False)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class _DenseBlock(nn.Module):
    """Dense block composed of multiple dense layers.

    Args:
        num_layers: Number of dense layers.
        in_channels: Input channel count.
        growth_rate: Growth rate per layer. Defaults to 32.
        bn_size: Bottleneck size. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.0.
    """

    def __init__(
        self, num_layers: int, in_channels: int, growth_rate: int = 32, bn_size: int = 4, dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(_DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _Transition(nn.Module):
    """Transition layer between dense blocks (compression + downsampling).

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class DenseNetBiModal(nn.Module):
    """DenseNet-based bimodal fusion baseline.

    Concatenates face and fingerprint as 6-channel input and processes
    through a DenseNet backbone. This treats multimodal data as a single
    unified input, allowing the network to learn joint features from
    the earliest layers.

    Args:
        growth_rate: DenseNet growth rate. Defaults to 32.
        block_config: Number of layers in each dense block.
            Defaults to (6, 12, 24, 16).
        num_init_features: Initial feature maps. Defaults to 64.
        bn_size: Bottleneck multiplier. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.1.
        feature_dim: Final feature dimension. Defaults to 512.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, ...] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        dropout: float = 0.1,
        feature_dim: int = 512,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(6, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, growth_rate, bn_size, dropout)
            self.features.add_module(f"denseblock{i+1}", block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize DenseNet weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        face_img: Tensor,
        fp_img: Tensor,
        face_quality: Union[Tensor, float],
        fp_quality: Union[Tensor, float],
        missing_mask: Tensor,
    ) -> Tuple[Tensor, None, None]:
        """Forward pass of DenseNet bimodal baseline.

        Args:
            face_img: Face images (N, 3, H, W).
            fp_img: Fingerprint images (N, 3, H, W).
            face_quality: Face quality scores (N,) or (N, 1).
            fp_quality: Fingerprint quality scores (N,) or (N, 1).
            missing_mask: Missing modality mask (N, 2).

        Returns:
            Tuple of (score, None, None).
        """
        face_missing = missing_mask[:, 0].bool()
        fp_missing = missing_mask[:, 1].bool()
        face_input = face_img.clone()
        fp_input = fp_img.clone()
        face_input[face_missing] = 0.0
        fp_input[fp_missing] = 0.0
        combined = torch.cat([face_input, fp_input], dim=1)
        features = self.features(combined)
        features = F.relu(features)
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)
        score = self.classifier(features).squeeze(-1)
        return score, None, None


class _BaselineEncoder(nn.Module):
    """Lightweight CNN encoder for baseline models.

    Self-contained encoder used by TransformerFusionNoUncertainty
    to avoid external dependencies.

    Args:
        in_channels: Input channels (3 for RGB).
        feature_dim: Output feature dimension. Defaults to 512.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(self, in_channels: int = 3, feature_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _BaselineTransformerLayer(nn.Module):
    """Single transformer fusion layer for baselines.

    Self-contained cross-attention layer used by TransformerFusionNoUncertainty.

    Args:
        embed_dim: Embedding dimension. Defaults to 512.
        num_heads: Number of attention heads. Defaults to 8.
        ff_dim: Feed-forward dimension. Defaults to 2048.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor, quality_scores: Optional[Tensor] = None) -> Tensor:
        attn_out, _ = self.cross_attn(query, key, value, need_weights=False)
        out = self.norm1(query + attn_out)
        out = self.norm2(out + self.ffn(out))
        return out


class TransformerFusionNoUncertainty(nn.Module):
    """Transformer fusion WITHOUT uncertainty quantification.

    Uses a transformer fusion architecture similar to the full UFM-Transformer
    but removes the uncertainty quantification head. Used to isolate and
    measure the contribution of uncertainty estimation.

    This implementation is fully self-contained and does not depend on
    model-specific internal classes.

    Args:
        feature_dim: Feature dimension. Defaults to 512.
        num_fusion_layers: Number of transformer layers. Defaults to 4.
        num_heads: Number of attention heads. Defaults to 8.
        ff_dim: Feed-forward dimension. Defaults to 2048.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(
        self, feature_dim: int = 512, num_fusion_layers: int = 4,
        num_heads: int = 8, ff_dim: int = 2048, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.face_encoder = _BaselineEncoder(3, feature_dim, dropout)
        self.fp_encoder = _BaselineEncoder(3, feature_dim, dropout)
        self.face_mask_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.fp_mask_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.quality_embed = nn.Sequential(
            nn.Linear(1, feature_dim // 4), nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, feature_dim),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.fusion_layers = nn.ModuleList([
            _BaselineTransformerLayer(feature_dim, num_heads, ff_dim, dropout)
            for _ in range(num_fusion_layers)
        ])
        self.fusion_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(feature_dim // 2, 1), nn.Sigmoid(),
        )
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def _encode_modality(
        self, encoder: nn.Module, images: Tensor, mask_token: nn.Parameter, missing_flags: Tensor
    ) -> Tensor:
        N = images.shape[0]
        features = torch.zeros(N, self.feature_dim, device=images.device, dtype=images.dtype)
        present_mask = ~missing_flags
        if present_mask.any():
            features[present_mask] = encoder(images[present_mask])
        if (~present_mask).any():
            missing_features = mask_token.expand((~present_mask).sum(), -1, -1).squeeze(1)
            features[~present_mask] = missing_features
        return features

    def forward(
        self, face_img: Tensor, fp_img: Tensor,
        face_quality: Union[Tensor, float], fp_quality: Union[Tensor, float],
        missing_mask: Tensor,
    ) -> Tuple[Tensor, None, None]:
        """Forward pass without uncertainty estimation.

        Args:
            face_img: Face images (N, 3, H, W).
            fp_img: Fingerprint images (N, 3, H, W).
            face_quality: Face quality scores (N,) or (N, 1).
            fp_quality: Fingerprint quality scores (N,) or (N, 1).
            missing_mask: Missing modality mask (N, 2).

        Returns:
            Tuple of (score, None, None).
        """
        N = face_img.shape[0]
        device = face_img.device
        if isinstance(face_quality, (int, float)):
            face_quality = torch.full((N,), face_quality, device=device, dtype=torch.float32)
        if isinstance(fp_quality, (int, float)):
            fp_quality = torch.full((N,), fp_quality, device=device, dtype=torch.float32)
        if face_quality.dim() == 0:
            face_quality = face_quality.unsqueeze(0).expand(N)
        if fp_quality.dim() == 0:
            fp_quality = fp_quality.unsqueeze(0).expand(N)
        if face_quality.dim() == 1:
            face_quality = face_quality.unsqueeze(1)
        if fp_quality.dim() == 1:
            fp_quality = fp_quality.unsqueeze(1)

        face_missing = missing_mask[:, 0].bool()
        fp_missing = missing_mask[:, 1].bool()
        face_features = self._encode_modality(self.face_encoder, face_img, self.face_mask_token, face_missing)
        fp_features = self._encode_modality(self.fp_encoder, fp_img, self.fp_mask_token, fp_missing)
        face_features = face_features + self.quality_embed(face_quality)
        fp_features = fp_features + self.quality_embed(fp_quality)
        face_features = face_features.unsqueeze(1)
        fp_features = fp_features.unsqueeze(1)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        tokens = torch.cat([cls_tokens, face_features, fp_features], dim=1)
        cls_quality = torch.ones(N, 1, device=device, dtype=torch.float32)
        quality_scores = torch.cat([cls_quality, face_quality, fp_quality], dim=1)
        fused = tokens
        for layer in self.fusion_layers:
            fused = layer(fused, fused, fused, quality_scores)
        cls_repr = self.fusion_norm(fused[:, 0, :])
        score = self.classifier(cls_repr).squeeze(-1)
        return score, None, None


class MDRLFusion(nn.Module):
    """Multi-Deep Representation Learning (MDRL) fusion baseline.

    Extracts features at multiple hierarchical levels from each modality
    and fuses them progressively with quality-aware gating and scale
    attention. This mimics approaches that leverage multiple layers of
    a deep network for richer representations.

    Args:
        feature_dim: Base feature dimension. Defaults to 256.
        num_scales: Number of feature extraction scales. Defaults to 4.
        dropout: Dropout probability. Defaults to 0.1.
    """

    def __init__(self, feature_dim: int = 256, num_scales: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales

        face_dims = [64, 128, 256, feature_dim]
        fp_dims = [64, 128, 256, feature_dim]

        def _make_ms_encoder(dims: List[int]) -> nn.ModuleList:
            layers = nn.ModuleList()
            in_ch = 3
            for dim in dims:
                block = nn.Sequential(
                    nn.Conv2d(in_ch, dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim, dim, 3, stride=1, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.ReLU(inplace=True),
                )
                layers.append(block)
                in_ch = dim
            return layers

        self.face_encoders = _make_ms_encoder(face_dims)
        self.fp_encoders = _make_ms_encoder(fp_dims)

        self.face_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_scales)])
        self.fp_pools = nn.ModuleList([nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_scales)])

        self.fusion_projectors = nn.ModuleList()
        for i in range(num_scales):
            proj = nn.Sequential(
                nn.Linear(face_dims[i] + fp_dims[i], feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.fusion_projectors.append(proj)

        self.quality_gates = nn.ModuleList([
            nn.Sequential(nn.Linear(2, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(num_scales)
        ])

        self.scale_attention = nn.Sequential(
            nn.Linear(feature_dim * num_scales, num_scales),
            nn.Softmax(dim=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        face_img: Tensor,
        fp_img: Tensor,
        face_quality: Union[Tensor, float],
        fp_quality: Union[Tensor, float],
        missing_mask: Tensor,
    ) -> Tuple[Tensor, None, None]:
        """Forward pass of MDRL fusion baseline.

        Args:
            face_img: Face images (N, 3, H, W).
            fp_img: Fingerprint images (N, 3, H, W).
            face_quality: Face quality scores (N,) or (N, 1).
            fp_quality: Fingerprint quality scores (N,) or (N, 1).
            missing_mask: Missing modality mask (N, 2).

        Returns:
            Tuple of (score, None, None).
        """
        N = face_img.shape[0]
        device = face_img.device
        if isinstance(face_quality, (int, float)):
            face_quality = torch.full((N,), face_quality, device=device, dtype=torch.float32)
        if isinstance(fp_quality, (int, float)):
            fp_quality = torch.full((N,), fp_quality, device=device, dtype=torch.float32)
        if face_quality.dim() == 0:
            face_quality = face_quality.unsqueeze(0).expand(N)
        if fp_quality.dim() == 0:
            fp_quality = fp_quality.unsqueeze(0).expand(N)
        if face_quality.dim() == 1:
            face_quality = face_quality.unsqueeze(1)
        if fp_quality.dim() == 1:
            fp_quality = fp_quality.unsqueeze(1)

        face_missing = missing_mask[:, 0].bool()
        fp_missing = missing_mask[:, 1].bool()
        face_input = face_img.clone()
        fp_input = fp_img.clone()
        face_input[face_missing] = 0.0
        fp_input[fp_missing] = 0.0

        face_x, fp_x = face_input, fp_input
        scale_features: List[Tensor] = []
        quality_input = torch.cat([face_quality, fp_quality], dim=1)

        for i in range(self.num_scales):
            face_x = self.face_encoders[i](face_x)
            fp_x = self.fp_encoders[i](fp_x)
            face_feat = self.face_pools[i](face_x).view(N, -1)
            fp_feat = self.fp_pools[i](fp_x).view(N, -1)
            combined = torch.cat([face_feat, fp_feat], dim=1)
            fused = self.fusion_projectors[i](combined)
            gate = self.quality_gates[i](quality_input)
            fused = fused * gate
            scale_features.append(fused)

        all_scales = torch.cat(scale_features, dim=1)
        scale_weights = self.scale_attention(all_scales)
        stacked = torch.stack(scale_features, dim=1)
        weighted = (scale_weights.unsqueeze(-1) * stacked).sum(dim=1)
        score = self.classifier(weighted).squeeze(-1)
        return score, None, None


# =============================================================================
# 3. EVALUATION PROTOCOL
# =============================================================================


def _generate_pair_scores_from_embeddings(
    scores: np.ndarray,
    subject_ids: np.ndarray,
    max_pairs_per_subject: int = 20,
    max_impostor_pairs: int = 500,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate genuine and impostor pair scores from individual sample scores.

    For datasets that return individual samples rather than pairs, this
    function generates verification pairs by matching samples from the
    same subject (genuine) and different subjects (impostor).

    Args:
        scores: Individual sample scores from the model.
        subject_ids: Subject ID for each sample.
        max_pairs_per_subject: Maximum genuine pairs per subject.
        max_impostor_pairs: Maximum impostor pairs total.
        seed: Random seed for pair selection.

    Returns:
        Tuple of (scores_genuine, scores_impostor).
    """
    np.random.seed(seed)

    subject_samples: Dict[int, List[int]] = {}
    for idx, sid in enumerate(subject_ids):
        sid_int = int(sid)
        if sid_int not in subject_samples:
            subject_samples[sid_int] = []
        subject_samples[sid_int].append(idx)

    subjects = list(subject_samples.keys())

    genuine_scores: List[float] = []
    for sid in subjects:
        samples = subject_samples[sid]
        if len(samples) < 2:
            continue
        n_pairs = min(max_pairs_per_subject, len(samples) * (len(samples) - 1) // 2)
        pairs_created = 0
        attempts = 0
        used_pairs = set()
        while pairs_created < n_pairs and attempts < n_pairs * 100:
            attempts += 1
            i, j = np.random.choice(samples, 2, replace=False)
            pair_key = tuple(sorted([int(i), int(j)]))
            if pair_key not in used_pairs:
                used_pairs.add(pair_key)
                genuine_scores.append(float((scores[i] + scores[j]) / 2.0))
                pairs_created += 1

    impostor_scores: List[float] = []
    impostor_created = 0
    attempts = 0
    used_impostor = set()
    while impostor_created < max_impostor_pairs and attempts < max_impostor_pairs * 100:
        attempts += 1
        if len(subjects) < 2:
            break
        subj_a, subj_b = np.random.choice(subjects, 2, replace=False)
        sample_a = int(np.random.choice(subject_samples[int(subj_a)]))
        sample_b = int(np.random.choice(subject_samples[int(subj_b)]))
        pair_key = tuple(sorted([sample_a, sample_b]))
        if pair_key not in used_impostor:
            used_impostor.add(pair_key)
            impostor_scores.append(float((scores[sample_a] + scores[sample_b]) / 2.0))
            impostor_created += 1

    return np.array(genuine_scores, dtype=np.float64), np.array(impostor_scores, dtype=np.float64)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a model on the test set and compute all metrics.

    This is the primary evaluation function that runs the model over the
    test dataset, collects genuine and impostor scores, and computes a
    comprehensive set of biometric evaluation metrics.

    Args:
        model: Biometric verification model (UFM-Transformer or baseline).
            Must implement the unified forward interface.
        test_loader: DataLoader providing test data.
        device: Torch device for computation.

    Returns:
        Tuple of (metrics_dict, scores_genuine, scores_impostor, predictions):
            - metrics_dict: Dictionary with all computed metrics.
            - scores_genuine: Genuine pair similarity scores.
            - scores_impostor: Impostor pair similarity scores.
            - predictions: Binary predictions for all test samples.

    Example:
        >>> metrics, gen_scores, imp_scores, preds = evaluate_model(model, test_loader, device)
        >>> print(f"EER = {metrics['eer']:.2f}%")
    """
    model.eval()
    all_scores: List[float] = []
    all_labels: List[int] = []
    all_predictions: List[int] = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) < 5:
                continue
            face, fp, subject_ids_or_labels, face_q, fp_q = batch
            face = face.to(device, non_blocking=True)
            fp = fp.to(device, non_blocking=True)

            # Parse quality
            if isinstance(face_q, torch.Tensor):
                face_q_t = face_q.to(device, non_blocking=True)
                if face_q_t.dim() == 0:
                    face_q_t = face_q_t.unsqueeze(0).expand(face.shape[0])
            else:
                face_q_t = torch.ones(face.shape[0], device=device) * 0.8
            if isinstance(fp_q, torch.Tensor):
                fp_q_t = fp_q.to(device, non_blocking=True)
                if fp_q_t.dim() == 0:
                    fp_q_t = fp_q_t.unsqueeze(0).expand(fp.shape[0])
            else:
                fp_q_t = torch.ones(fp.shape[0], device=device) * 0.8

            missing = torch.zeros(face.shape[0], 2, device=device)
            scores, _, _ = model(face, fp, face_q_t, fp_q_t, missing)
            scores_np = scores.cpu().numpy()
            labels_np = subject_ids_or_labels.cpu().numpy().astype(int)

            all_scores.extend(scores_np.tolist())
            all_labels.extend(labels_np.tolist())

    all_scores_np = np.array(all_scores, dtype=np.float64)
    all_labels_np = np.array(all_labels, dtype=int)

    unique_labels = np.unique(all_labels_np)
    if len(unique_labels) > max(2, len(all_labels_np) // 10):
        scores_genuine, scores_impostor = _generate_pair_scores_from_embeddings(
            all_scores_np, all_labels_np
        )
    else:
        scores_genuine = all_scores_np[all_labels_np == 1]
        scores_impostor = all_scores_np[all_labels_np == 0]

    if len(scores_genuine) == 0 or len(scores_impostor) == 0:
        warnings.warn("Empty genuine or impostor score arrays. Returning default metrics.")
        return {
            "eer": 50.0, "eer_threshold": 0.5, "tar_at_0_001": 50.0,
            "tar_at_0_01": 50.0, "auc": 50.0, "fnmr_at_eer": 50.0,
            "fmr_at_eer": 50.0, "ece": 50.0, "mce": 50.0,
        }, scores_genuine, scores_impostor, np.array([])

    metrics = compute_all_metrics(scores_genuine, scores_impostor)
    predictions = (all_scores_np >= 0.5).astype(int)
    return metrics, scores_genuine, scores_impostor, predictions


# ---- Robustness Evaluation ----

class _GaussianBlurDegradation:
    """Apply Gaussian blur degradation.

    Args:
        sigma: Standard deviation of the Gaussian kernel.
    """

    def __init__(self, sigma: float = 2.0) -> None:
        self.sigma = sigma

    def __call__(self, img: Tensor) -> Tensor:
        img_pil = TF.to_pil_image(img)
        img_pil = img_pil.filter(T.functional.ImageFilter.GaussianBlur(radius=self.sigma))
        return TF.to_tensor(img_pil)


class _GaussianNoiseDegradation:
    """Add Gaussian noise degradation.

    Args:
        std: Standard deviation of the noise.
    """

    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def __call__(self, img: Tensor) -> Tensor:
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0.0, 1.0)


class _OcclusionDegradation:
    """Apply block occlusion degradation.

    Args:
        occlusion_ratio: Fraction of image to occlude.
        fill_value: Value to fill occluded region with. Defaults to 0.
    """

    def __init__(self, occlusion_ratio: float = 0.3, fill_value: float = 0.0) -> None:
        self.occlusion_ratio = occlusion_ratio
        self.fill_value = fill_value

    def __call__(self, img: Tensor) -> Tensor:
        _, h, w = img.shape
        occ_h = int(self.occlusion_ratio * h)
        occ_w = int(self.occlusion_ratio * w)
        top = np.random.randint(0, max(1, h - occ_h))
        left = np.random.randint(0, max(1, w - occ_w))
        img_out = img.clone()
        img_out[:, top : top + occ_h, left : left + occ_w] = self.fill_value
        return img_out


class _MinutiaeDropoutDegradation:
    """Apply minutiae dropout for fingerprint images.

    Args:
        dropout_ratio: Fraction of regions to drop.
        num_regions: Number of circular dropout regions.
        radius: Radius of dropout regions in pixels.
    """

    def __init__(self, dropout_ratio: float = 0.5, num_regions: int = 15, radius: int = 6) -> None:
        self.num_regions = int(num_regions * dropout_ratio)
        self.radius = radius

    def __call__(self, img: Tensor) -> Tensor:
        c, h, w = img.shape
        img_out = img.clone()
        for _ in range(self.num_regions):
            cy = np.random.randint(self.radius, max(self.radius + 1, h - self.radius))
            cx = np.random.randint(self.radius, max(self.radius + 1, w - self.radius))
            y_grid, x_grid = torch.meshgrid(
                torch.arange(h, dtype=torch.float32),
                torch.arange(w, dtype=torch.float32),
                indexing="ij",
            )
            dist = torch.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
            mask = dist <= self.radius
            img_out[:, mask] = 0.0
        return img_out


def evaluate_robustness(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model robustness under various degradation conditions.

    Tests the model's performance when inputs are degraded in controlled
    ways that simulate real-world conditions.

    Degradation conditions tested:
        - Clean: No degradation (baseline)
        - Face Gaussian blur (sigma=2): Mild face blur
        - Face Gaussian blur (sigma=4): Severe face blur
        - Face noise (std=0.05): Sensor noise on face
        - Face occlusion (30%): Partial face occlusion
        - Fingerprint blur (sigma=2): Mild fingerprint blur
        - Fingerprint noise (std=0.05): Sensor noise on fingerprint
        - Fingerprint minutiae dropout (50%): Missing fingerprint features
        - Face only: Only face modality available
        - Fingerprint only: Only fingerprint modality available

    Args:
        model: Biometric verification model.
        test_loader: DataLoader providing test data.
        device: Torch device for computation.

    Returns:
        Dictionary mapping each condition name to its metrics dictionary.
        Format: {condition_name: {metric_name: value, ...}, ...}

    Example:
        >>> robustness = evaluate_robustness(model, test_loader, device)
        >>> print(robustness["Face only"]["eer"])
    """
    degradations = {
        "Clean": {"face": None, "fp": None, "missing": [0, 0]},
        "Face blur (sigma=2)": {"face": _GaussianBlurDegradation(2.0), "fp": None, "missing": [0, 0]},
        "Face blur (sigma=4)": {"face": _GaussianBlurDegradation(4.0), "fp": None, "missing": [0, 0]},
        "Face noise (std=0.05)": {"face": _GaussianNoiseDegradation(0.05), "fp": None, "missing": [0, 0]},
        "Face occlusion (30%)": {"face": _OcclusionDegradation(0.30), "fp": None, "missing": [0, 0]},
        "FP blur (sigma=2)": {"face": None, "fp": _GaussianBlurDegradation(2.0), "missing": [0, 0]},
        "FP noise (std=0.05)": {"face": None, "fp": _GaussianNoiseDegradation(0.05), "missing": [0, 0]},
        "FP minutiae dropout (50%)": {"face": None, "fp": _MinutiaeDropoutDegradation(0.5), "missing": [0, 0]},
        "Face only": {"face": None, "fp": None, "missing": [0, 1]},
        "Fingerprint only": {"face": None, "fp": None, "missing": [1, 0]},
    }

    results: Dict[str, Dict[str, float]] = {}
    model.eval()

    for condition_name, degradation in degradations.items():
        all_scores: List[float] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) < 5:
                    continue
                face, fp, subject_ids_or_labels, face_q, fp_q = batch
                face = face.to(device, non_blocking=True)
                fp = fp.to(device, non_blocking=True)

                if isinstance(face_q, torch.Tensor):
                    face_q_t = face_q.to(device, non_blocking=True)
                    if face_q_t.dim() == 0:
                        face_q_t = face_q_t.unsqueeze(0).expand(face.shape[0])
                else:
                    face_q_t = torch.ones(face.shape[0], device=device) * 0.8
                if isinstance(fp_q, torch.Tensor):
                    fp_q_t = fp_q.to(device, non_blocking=True)
                    if fp_q_t.dim() == 0:
                        fp_q_t = fp_q_t.unsqueeze(0).expand(fp.shape[0])
                else:
                    fp_q_t = torch.ones(fp.shape[0], device=device) * 0.8

                face_degraded = face.clone()
                fp_degraded = fp.clone()
                if degradation["face"] is not None:
                    face_degraded = torch.stack([degradation["face"](f.cpu()).to(device) for f in face])
                if degradation["fp"] is not None:
                    fp_degraded = torch.stack([degradation["fp"](f.cpu()).to(device) for f in fp])

                missing = torch.tensor(
                    [degradation["missing"]] * face.shape[0],
                    device=device, dtype=torch.float32,
                )

                scores, _, _ = model(face_degraded, fp_degraded, face_q_t, fp_q_t, missing)
                scores_np = scores.cpu().numpy()
                labels_np = subject_ids_or_labels.cpu().numpy().astype(int)
                all_scores.extend(scores_np.tolist())
                all_labels.extend(labels_np.tolist())

        all_scores_np = np.array(all_scores, dtype=np.float64)
        all_labels_np = np.array(all_labels, dtype=int)

        unique_labels = np.unique(all_labels_np)
        if len(unique_labels) > max(2, len(all_labels_np) // 10):
            scores_gen, scores_imp = _generate_pair_scores_from_embeddings(all_scores_np, all_labels_np)
        else:
            scores_gen = all_scores_np[all_labels_np == 1]
            scores_imp = all_scores_np[all_labels_np == 0]

        if len(scores_gen) == 0 or len(scores_imp) == 0:
            warnings.warn(f"Empty scores for condition '{condition_name}'. Skipping.")
            continue

        condition_metrics = compute_all_metrics(scores_gen, scores_imp)
        results[condition_name] = condition_metrics

    return results


# ---- Uncertainty Calibration Evaluation ----

def evaluate_uncertainty_calibration(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate uncertainty calibration of the model.

    For models with uncertainty quantification, analyzes reliability
    diagram data, confidence vs accuracy, and uncertainty distribution
    for correct vs incorrect predictions.

    For models without uncertainty (baselines), only reliability
    diagram and confidence analysis are computed.

    Args:
        model: Biometric verification model.
        test_loader: DataLoader providing test data.
        device: Torch device for computation.

    Returns:
        Dictionary containing calibration data:
            - confidences: Predicted confidence scores.
            - accuracies: Binary correctness labels.
            - correct_uncertainties: Uncertainty for correct predictions.
            - incorrect_uncertainties: Uncertainty for incorrect predictions.
            - reliability_bins: Bin edges for reliability diagram.
            - reliability_accuracies: Per-bin accuracy values.
            - reliability_confidences: Per-bin confidence values.

    Example:
        >>> cal_data = evaluate_uncertainty_calibration(model, test_loader, device)
        >>> plot_reliability_diagram(cal_data["reliability_confidences"],
        ...                          cal_data["reliability_accuracies"], "reliability.pdf")
    """
    model.eval()
    all_confidences: List[float] = []
    all_accuracies: List[int] = []
    correct_uncertainties: List[float] = []
    incorrect_uncertainties: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) < 5:
                continue
            face, fp, subject_ids_or_labels, face_q, fp_q = batch
            face = face.to(device, non_blocking=True)
            fp = fp.to(device, non_blocking=True)

            if isinstance(face_q, torch.Tensor):
                face_q_t = face_q.to(device, non_blocking=True)
                if face_q_t.dim() == 0:
                    face_q_t = face_q_t.unsqueeze(0).expand(face.shape[0])
            else:
                face_q_t = torch.ones(face.shape[0], device=device) * 0.8
            if isinstance(fp_q, torch.Tensor):
                fp_q_t = fp_q.to(device, non_blocking=True)
                if fp_q_t.dim() == 0:
                    fp_q_t = fp_q_t.unsqueeze(0).expand(fp.shape[0])
            else:
                fp_q_t = torch.ones(fp.shape[0], device=device) * 0.8

            missing = torch.zeros(face.shape[0], 2, device=device)
            scores, aleatoric, epistemic = model(face, fp, face_q_t, fp_q_t, missing)
            scores_np = scores.cpu().numpy()
            labels_np = subject_ids_or_labels.cpu().numpy().astype(int)

            for i in range(len(scores_np)):
                confidence = float(scores_np[i])
                label = int(labels_np[i])
                prediction = 1 if confidence >= 0.5 else 0
                correct = int(prediction == label) if label in (0, 1) else int(confidence > 0.5)
                all_confidences.append(confidence)
                all_accuracies.append(correct)

                if aleatoric is not None:
                    unc = float(torch.exp(aleatoric[i]).cpu().item())
                    if correct:
                        correct_uncertainties.append(unc)
                    else:
                        incorrect_uncertainties.append(unc)

    confidences_np = np.array(all_confidences)
    accuracies_np = np.array(all_accuracies)

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_confs, bin_accs, bin_counts = [], [], []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences_np >= lower) & (confidences_np <= upper)
        else:
            in_bin = (confidences_np >= lower) & (confidences_np < upper)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_confs.append(float(np.mean(confidences_np[in_bin])))
            bin_accs.append(float(np.mean(accuracies_np[in_bin])))
        else:
            bin_confs.append(float(bin_centers[i]))
            bin_accs.append(0.0)
        bin_counts.append(count)

    return {
        "confidences": confidences_np,
        "accuracies": accuracies_np,
        "correct_uncertainties": np.array(correct_uncertainties, dtype=np.float64),
        "incorrect_uncertainties": np.array(incorrect_uncertainties, dtype=np.float64),
        "reliability_bins": bin_edges,
        "reliability_accuracies": np.array(bin_accs),
        "reliability_confidences": np.array(bin_confs),
        "reliability_counts": np.array(bin_counts),
    }


# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================
# All functions save figures as publication-quality PDFs with bbox_inches='tight'.


def plot_roc_curve(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
    save_path: str,
) -> None:
    """Plot the Receiver Operating Characteristic (ROC) curve and save as PDF.

    The ROC curve plots the True Acceptance Rate (TAR = 1 - FNMR) against
    the False Acceptance Rate (FAR = FMR) at various operating thresholds.
    A perfect classifier achieves 100% TAR at 0% FAR.

    Args:
        scores_genuine: Genuine pair similarity scores.
        scores_impostor: Impostor pair similarity scores.
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> plot_roc_curve(genuine_scores, impostor_scores, "./figures/roc.pdf")
    """
    all_scores = np.concatenate([scores_genuine, scores_impostor])
    all_labels = np.concatenate([
        np.ones(len(scores_genuine)),
        np.zeros(len(scores_impostor)),
    ])
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = float(auc(fpr, tpr))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr * 100, tpr * 100, color=C_GEN, linewidth=2.5,
            label=f"ROC Curve (AUC = {roc_auc * 100:.2f}%)")
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1.5,
            label="Random Classifier (AUC = 50%)")
    ax.set_xlabel("False Acceptance Rate (%)", fontweight="bold")
    ax.set_ylabel("True Acceptance Rate (%)", fontweight="bold")
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontweight="bold", fontsize=14)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.15, f"AUC = {roc_auc * 100:.2f}%",
            transform=ax.transAxes, fontsize=14, fontweight="bold",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_det_curve(
    fmr: np.ndarray,
    fnmr: np.ndarray,
    save_path: str,
) -> None:
    """Plot the Detection Error Tradeoff (DET) curve and save as PDF.

    The DET curve plots FNMR vs FMR on normal deviate (probit) scales.
    This spreads out the region near the EER for better visualization
    than the ROC curve. It is the preferred plot in biometric literature.

    Args:
        fmr: False Match Rate values (0-100%).
        fnmr: False Non-Match Rate values (0-100%).
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> fmr_arr, fnmr_arr, _ = compute_det_curve(genuine_scores, impostor_scores)
        >>> plot_det_curve(fmr_arr, fnmr_arr, "./figures/det.pdf")
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    fmr_clipped = np.clip(fmr / 100.0, 1e-6, 1 - 1e-6)
    fnmr_clipped = np.clip(fnmr / 100.0, 1e-6, 1 - 1e-6)

    fmr_probit = norm.ppf(fmr_clipped)
    fnmr_probit = norm.ppf(fnmr_clipped)

    ax.plot(fmr_probit, fnmr_probit, color=C_GEN, linewidth=2.5, label="DET Curve")
    eer_line_x = np.linspace(fmr_probit.min(), fmr_probit.max(), 100)
    ax.plot(eer_line_x, eer_line_x, color=C_ACC, linestyle=":", linewidth=1.5,
            label="EER Line (FMR = FNMR)")

    ticks = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 40, 60, 80, 95, 99]
    ticks = [t for t in ticks if 0.01 <= t <= 99.0]
    tick_positions = norm.ppf(np.array(ticks) / 100.0)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{t:.2f}" if t < 1 else f"{t:.1f}" for t in ticks], rotation=45)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{t:.2f}" if t < 1 else f"{t:.1f}" for t in ticks])

    ax.set_xlabel("False Match Rate (%)", fontweight="bold")
    ax.set_ylabel("False Non-Match Rate (%)", fontweight="bold")
    ax.set_title("Detection Error Tradeoff (DET) Curve", fontweight="bold", fontsize=14)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_score_distributions(
    scores_genuine: np.ndarray,
    scores_impostor: np.ndarray,
    save_path: str,
) -> None:
    """Plot overlapping histograms of genuine and impostor score distributions.

    This visualization shows the separation between genuine and impostor
    score distributions. Good biometric systems exhibit minimal overlap.

    Args:
        scores_genuine: Genuine pair similarity scores.
        scores_impostor: Impostor pair similarity scores.
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> plot_score_distributions(genuine_scores, impostor_scores, "./figures/dists.pdf")
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, 1, 51)

    ax.hist(scores_impostor, bins=bins, alpha=0.7, color=C_IMP,
            label=f"Impostor (N={len(scores_impostor):,})", density=True,
            edgecolor="white", linewidth=0.5)
    ax.hist(scores_genuine, bins=bins, alpha=0.7, color=C_GEN,
            label=f"Genuine (N={len(scores_genuine):,})", density=True,
            edgecolor="white", linewidth=0.5)

    try:
        eer, threshold = compute_eer(scores_genuine, scores_impostor)
        ax.axvline(threshold, color=C_ACC, linestyle="--", linewidth=2,
                   label=f"EER Threshold = {threshold:.4f}")
    except ValueError:
        pass

    ax.set_xlabel("Similarity Score", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title("Genuine vs. Impostor Score Distributions", fontweight="bold", fontsize=14)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: str,
) -> None:
    """Plot a confusion matrix heatmap and save as PDF.

    Args:
        predictions: Binary predictions (0 or 1).
        labels: Ground truth labels (0 or 1).
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> plot_confusion_matrix(preds, labels, "./figures/confusion.pdf")
    """
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                xticklabels=["Impostor (0)", "Genuine (1)"],
                yticklabels=["Impostor (0)", "Genuine (1)"],
                linewidths=1, linecolor="white", annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_violin_scores(
    scores_by_condition: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str,
) -> None:
    """Plot violin plots of score distributions per degradation condition.

    Compares genuine and impostor score distributions across different
    degradation conditions, showing how each degradation type affects scores.

    Args:
        scores_by_condition: Dictionary mapping condition names to tuples of
            (scores_genuine, scores_impostor).
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> conditions = {"Clean": (c_gen, c_imp), "Blur": (b_gen, b_imp)}
        >>> plot_violin_scores(conditions, "./figures/violin.pdf")
    """
    import pandas as pd

    fig_width = max(10, len(scores_by_condition) * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    data_rows = []
    for condition_name, (genuine, impostor) in scores_by_condition.items():
        for s in genuine:
            data_rows.append({"Score": s, "Group": f"{condition_name}\n(Genuine)", "Type": "Genuine"})
        for s in impostor:
            data_rows.append({"Score": s, "Group": f"{condition_name}\n(Impostor)", "Type": "Impostor"})

    df = pd.DataFrame(data_rows)

    # Create alternating color palette
    palette = {}
    for condition_name in scores_by_condition:
        palette[f"{condition_name}\n(Genuine)"] = C_GEN
        palette[f"{condition_name}\n(Impostor)"] = C_IMP

    sns.violinplot(data=df, x="Group", y="Score", ax=ax, palette=palette, inner="box")

    ax.set_xlabel("Condition", fontweight="bold")
    ax.set_ylabel("Similarity Score", fontweight="bold")
    ax.set_title("Score Distributions by Condition", fontweight="bold", fontsize=14)
    ax.set_ylim([0, 1])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    save_path: str,
) -> None:
    """Plot a reliability diagram (confidence vs. accuracy) and save as PDF.

    A reliability diagram shows whether predicted confidence reflects true
    accuracy. A perfectly calibrated model follows the diagonal line.

    Args:
        confidences: Predicted confidence scores.
        accuracies: Binary correctness labels (1 = correct, 0 = incorrect).
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> plot_reliability_diagram(conf, acc, "./figures/reliability.pdf")
    """
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_confs, bin_accs, bin_counts = [], [], []

    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences >= lower) & (confidences < upper)
        count = int(np.sum(in_bin))
        if count > 0:
            bin_confs.append(float(np.mean(confidences[in_bin])))
            bin_accs.append(float(np.mean(accuracies[in_bin])))
        else:
            bin_confs.append(float(bin_centers[i]))
            bin_accs.append(0.0)
        bin_counts.append(count)

    ece, mce = compute_calibration_error(confidences, accuracies, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Perfect Calibration")

    gaps = np.array(bin_accs) - np.array(bin_confs)
    for i in range(n_bins):
        color = C_OK if gaps[i] >= 0 else C_ERR
        ax.bar(bin_centers[i], bin_accs[i], width=0.08, color=color, alpha=0.8, edgecolor="white")

    ax.plot(bin_centers, bin_accs, "o-", color=C_GEN, linewidth=2, markersize=8,
            label="Model Calibration")

    ax.set_xlabel("Mean Predicted Confidence", fontweight="bold")
    ax.set_ylabel("Fraction of Positives (Accuracy)", fontweight="bold")
    ax.set_title("Reliability Diagram", fontweight="bold", fontsize=14)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.text(0.95, 0.15, f"ECE = {ece:.2f}%\nMCE = {mce:.2f}%",
            transform=ax.transAxes, fontsize=12, fontweight="bold",
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_uncertainty_histogram(
    correct_uncertainties: np.ndarray,
    incorrect_uncertainties: np.ndarray,
    save_path: str,
) -> None:
    """Plot histogram comparing uncertainty for correct vs. incorrect predictions.

    A well-calibrated uncertainty model should show higher uncertainty for
    incorrect predictions, allowing the system to identify unreliable decisions.

    Args:
        correct_uncertainties: Uncertainty values for correct predictions.
        incorrect_uncertainties: Uncertainty values for incorrect predictions.
        save_path: Path to save the PDF figure.

    Returns:
        None. Figure is saved to disk.

    Example:
        >>> plot_uncertainty_histogram(correct_unc, incorrect_unc, "./figures/uncertainty.pdf")
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = 30

    if len(correct_uncertainties) > 0:
        ax.hist(correct_uncertainties, bins=bins, alpha=0.7, color=C_OK,
                label=f"Correct (N={len(correct_uncertainties)})", density=True,
                edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(correct_uncertainties)), color=C_OK, linestyle="--",
                   linewidth=2, label=f"Mean Correct = {np.mean(correct_uncertainties):.4f}")

    if len(incorrect_uncertainties) > 0:
        ax.hist(incorrect_uncertainties, bins=bins, alpha=0.7, color=C_ERR,
                label=f"Incorrect (N={len(incorrect_uncertainties)})", density=True,
                edgecolor="white", linewidth=0.5)
        ax.axvline(float(np.mean(incorrect_uncertainties)), color=C_ERR, linestyle="--",
                   linewidth=2, label=f"Mean Incorrect = {np.mean(incorrect_uncertainties):.4f}")

    ax.set_xlabel("Uncertainty (Aleatoric)", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title("Uncertainty Distribution: Correct vs. Incorrect Predictions",
                 fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# =============================================================================
# 5. LaTeX EXPORT FUNCTIONS
# =============================================================================


def export_metrics_latex(
    metrics_dict: Dict[str, float],
    save_path: str,
    model_name: str = "UFM-Transformer",
) -> None:
    """Export a single model's metrics as a LaTeX table.

    Generates a standalone LaTeX table snippet containing all evaluation
    metrics for a single model. Values are formatted with appropriate
    precision for publication, marked with \textcolor{red}{\textbf{VALUE}}
    and % INSERER VALEUR REELLE comments.

    Args:
        metrics_dict: Dictionary of metric names to values.
        save_path: Path to save the LaTeX file.
        model_name: Name of the model for the table caption. Defaults to "UFM-Transformer".

    Returns:
        None. LaTeX code is written to disk.

    Example:
        >>> metrics = compute_all_metrics(genuine_scores, impostor_scores)
        >>> export_metrics_latex(metrics, "./tables/metrics.tex", "UFM-Transformer")
    """
    metric_formatters = {
        "eer": ("EER", "{:.2f}"),
        "eer_threshold": ("EER Threshold", "{:.4f}"),
        "tar_at_0_001": ("TAR @ FAR=0.1%", "{:.2f}"),
        "tar_at_0_01": ("TAR @ FAR=1%", "{:.2f}"),
        "auc": ("AUC-ROC", "{:.2f}"),
        "fnmr_at_eer": ("FNMR @ EER", "{:.2f}"),
        "fmr_at_eer": ("FMR @ EER", "{:.2f}"),
        "ece": ("ECE (Calibration)", "{:.2f}"),
        "mce": ("MCE (Calibration)", "{:.2f}"),
    }

    latex = []
    latex.append("% LaTeX table: Evaluation metrics for " + model_name)
    latex.append("% Generated automatically by evaluate.py")
    latex.append("% INSERER VALEUR REELLE markers indicate where real values should be placed")
    latex.append("")
    latex.append(r"\begin{table}[t]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{" + f"Evaluation Metrics for {model_name}" + r"}")
    latex.append(r"    \label{tab:" + model_name.lower().replace(" ", "_").replace("-", "_") + "_metrics" + r"}")
    latex.append(r"    \begin{tabular}{lc}")
    latex.append(r"        \toprule")
    latex.append(r"        \textbf{Metric} & \textbf{Value} \\")
    latex.append(r"        \midrule")

    for key, (display_name, fmt) in metric_formatters.items():
        if key in metrics_dict:
            value_str = fmt.format(metrics_dict[key])
            latex.append(f"        {display_name} & \\textcolor{{red}}{{\\textbf{{{value_str}}}}} \\")
        else:
            latex.append(f"        {display_name} & \\textcolor{{red}}{{\\textbf{{--}}}} \\ % INSERER VALEUR REELLE: {key}")

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append(r"\end{table}")
    latex.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(latex))


def export_comparison_table_latex(
    all_metrics: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Export comparison table: UFM-Transformer vs. all baselines as LaTeX.

    Generates a comprehensive comparison table showing all models side-by-side
    with their key metrics. This is the primary table for publication.

    Args:
        all_metrics: Dictionary mapping model names to their metrics dictionaries.
        save_path: Path to save the LaTeX file.

    Returns:
        None. LaTeX code is written to disk.

    Example:
        >>> all_metrics = {"UFM-Transformer": ufm_metrics, "Concat-Fusion": concat_metrics}
        >>> export_comparison_table_latex(all_metrics, "./tables/comparison.tex")
    """
    display_metrics = [
        ("eer", "EER (%)"),
        ("tar_at_0_001", "TAR@0.1\\% FMR (%)"),
        ("tar_at_0_01", "TAR@1\\% FMR (%)"),
        ("auc", "AUC (%)"),
        ("ece", "ECE (%)"),
        ("mce", "MCE (%)"),
    ]

    model_names = list(all_metrics.keys())

    latex = []
    latex.append("% LaTeX comparison table: UFM-Transformer vs. Baselines")
    latex.append("% Generated automatically by evaluate.py")
    latex.append("% INSERER VALEUR REELLE markers indicate where real values should be inserted")
    latex.append("")
    latex.append(r"\begin{table*}[t]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Performance Comparison: UFM-Transformer vs. Baseline Methods}")
    latex.append(r"    \label{tab:model_comparison}")

    cols = "l" + "c" * len(model_names)
    latex.append(f"    \\begin{{tabular}}{{{cols}}}")
    latex.append(r"        \toprule")

    header = "        \\textbf{Metric}"
    for name in model_names:
        latex_name = name.replace("&", "\\&")
        header += f" & \\textbf{{{latex_name}}}"
    header += " \\\\"
    latex.append(header)
    latex.append(r"        \midrule")

    for metric_key, display_name in display_metrics:
        row = f"        {display_name}"
        for name in model_names:
            if name in all_metrics and metric_key in all_metrics[name]:
                value = all_metrics[name][metric_key]
                row += f" & \\textcolor{{red}}{{\\textbf{{{value:.2f}}}}}"
            else:
                row += r" & \textcolor{red}{\textbf{--}}"
        row += " \\\\"
        latex.append(row)

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append("")
    latex.append(r"    \begin{tablenotes}")
    latex.append(r"        \small")
    latex.append(r"        \item Note: Best results are shown in \textbf{bold}. ")
    latex.append(r"        EER = Equal Error Rate, TAR = True Acceptance Rate,")
    latex.append(r"        AUC = Area Under Curve, ECE = Expected Calibration Error,")
    latex.append(r"        MCE = Maximum Calibration Error.")
    latex.append(r"    \end{tablenotes}")
    latex.append(r"\end{table*}")
    latex.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(latex))


def export_robustness_table_latex(
    robustness_results: Dict[str, Dict[str, float]],
    save_path: str,
) -> None:
    """Export robustness evaluation results as a LaTeX table.

    Generates a table showing model performance under various degradation
    conditions, highlighting robustness to quality variations and missing
    modalities.

    Args:
        robustness_results: Dictionary from evaluate_robustness().
        save_path: Path to save the LaTeX file.

    Returns:
        None. LaTeX code is written to disk.

    Example:
        >>> robustness = evaluate_robustness(model, test_loader, device)
        >>> export_robustness_table_latex(robustness, "./tables/robustness.tex")
    """
    display_metrics = [
        ("eer", "EER (%)"),
        ("tar_at_0_001", "TAR@0.1\\% FMR"),
        ("tar_at_0_01", "TAR@1\\% FMR"),
        ("auc", "AUC (%)"),
    ]

    condition_names = list(robustness_results.keys())

    latex = []
    latex.append("% LaTeX robustness table: Performance under degradation conditions")
    latex.append("% Generated automatically by evaluate.py")
    latex.append("% INSERER VALEUR REELLE markers indicate where real values should be inserted")
    latex.append("")
    latex.append(r"\begin{table*}[t]")
    latex.append(r"    \centering")
    latex.append(r"    \caption{Robustness Evaluation: Performance Under Degradation Conditions}")
    latex.append(r"    \label{tab:robustness}")

    cols = "l" + "c" * len(condition_names)
    latex.append(f"    \\begin{{tabular}}{{{cols}}}")
    latex.append(r"        \toprule")

    header = "        \\textbf{Metric}"
    for cond in condition_names:
        escaped = cond.replace("%", "\\%").replace("&", "\\&")
        header += f" & \\textbf{{{escaped}}}"
    header += " \\\\"
    latex.append(header)
    latex.append(r"        \midrule")

    for metric_key, display_name in display_metrics:
        row = f"        {display_name}"
        for cond in condition_names:
            if cond in robustness_results and metric_key in robustness_results[cond]:
                value = robustness_results[cond][metric_key]
                row += f" & \\textcolor{{red}}{{\\textbf{{{value:.2f}}}}}"
            else:
                row += r" & \textcolor{red}{\textbf{--}}"
        row += " \\\\"
        latex.append(row)

    latex.append(r"        \bottomrule")
    latex.append(r"    \end{tabular}")
    latex.append("")
    latex.append(r"    \begin{tablenotes}")
    latex.append(r"        \small")
    latex.append(r"        \item Clean = No degradation (baseline). Face/FP only = Single modality.")
    latex.append(r"        \item Blur = Gaussian blur, Noise = Gaussian noise,")
    latex.append(r"        Occlusion = Block occlusion, Dropout = Minutiae dropout.")
    latex.append(r"    \end{tablenotes}")
    latex.append(r"\end{table*}")
    latex.append("")

    with open(save_path, "w") as f:
        f.write("\n".join(latex))


# =============================================================================
# 6. MAIN EVALUATION SCRIPT
# =============================================================================


def _setup_directories(output_dir: str) -> Tuple[str, str]:
    """Create output directory structure.

    Args:
        output_dir: Root output directory.

    Returns:
        Tuple of (figures_dir, tables_dir).
    """
    figures_dir = os.path.join(output_dir, "figures")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    return figures_dir, tables_dir


def _load_model(
    model_path: str,
    model_class: type,
    device: torch.device,
    **model_kwargs: Any,
) -> nn.Module:
    """Load a model from checkpoint file.

    Args:
        model_path: Path to the .pth checkpoint file.
        model_class: Model class to instantiate.
        device: Torch device.
        **model_kwargs: Keyword arguments for model constructor.

    Returns:
        Loaded model on the specified device.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        RuntimeError: If checkpoint loading fails.
    """
    model = model_class(**model_kwargs).to(device)

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print(f"  Loaded checkpoint from: {model_path}")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
            print("  Using randomly initialized weights.")
    else:
        print(f"  Checkpoint not found at {model_path}. Using randomly initialized weights.")

    model.eval()
    return model


def _create_baseline_models(device: torch.device) -> Dict[str, nn.Module]:
    """Create all baseline models with default configurations.

    Args:
        device: Torch device.

    Returns:
        Dictionary mapping baseline names to model instances.
    """
    baselines = {}
    baselines["Concat-Fusion"] = ConcatenationFusion(feature_dim=256, dropout=0.1).to(device)
    baselines["Score-Sum"] = ScoreSumFusion(feature_dim=256, dropout=0.1).to(device)
    baselines["DenseNet"] = DenseNetBiModal(
        growth_rate=32, block_config=(6, 12, 24, 16), dropout=0.1, feature_dim=512
    ).to(device)
    baselines["Transformer-NoUnc"] = TransformerFusionNoUncertainty(
        feature_dim=512, num_fusion_layers=4, num_heads=8, dropout=0.1
    ).to(device)
    baselines["MDRL"] = MDRLFusion(feature_dim=256, num_scales=4, dropout=0.1).to(device)
    return baselines


def main() -> None:
    """Run the complete evaluation pipeline.

    This function orchestrates the entire evaluation workflow:
        1. Parse command-line arguments.
        2. Load the test dataset.
        3. Load the UFM-Transformer model and all baselines.
        4. Evaluate all models and collect metrics.
        5. Generate all visualization plots as PDFs.
        6. Export LaTeX tables.
        7. Print summary comparison table to console.

    Command-line arguments:
        --model_path: Path to trained UFM-Transformer checkpoint (.pth).
        --dataset_path: Root directory of the test dataset.
        --output_dir: Directory for output figures and tables.
        --batch_size: Batch size for evaluation. Defaults to 32.
        --image_size: Image size for preprocessing. Defaults to 224.
        --num_workers: DataLoader worker processes. Defaults to 4.
        --seed: Random seed for reproducibility. Defaults to 42.
        --skip_baselines: Skip baseline model evaluation.
        --skip_robustness: Skip robustness evaluation.
        --skip_uncertainty: Skip uncertainty calibration evaluation.
        --device: Device to use (cuda/cpu). Defaults to auto.

    Example:
        $ python evaluate.py \\
            --model_path ./checkpoints/best_model.pth \\
            --dataset_path /path/to/data \\
            --output_dir ./results
    """
    parser = argparse.ArgumentParser(
        description="UFM-Transformer: Evaluation and Baseline Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation with all baselines
  python evaluate.py --model_path ./checkpoints/best_model.pth --dataset_path /data/biometric --output_dir ./results

  # Quick evaluation (skip baselines and robustness)
  python evaluate.py --model_path ./checkpoints/best_model.pth --dataset_path /data/biometric --output_dir ./results --skip_baselines --skip_robustness

  # CPU-only evaluation
  python evaluate.py --model_path ./checkpoints/best_model.pth --dataset_path /data/biometric --output_dir ./results --device cpu --batch_size 16
        """,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="./checkpoints/best_model.pth",
        help="Path to trained UFM-Transformer checkpoint (.pth file). "
             "If not found, a randomly initialized model is used.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root directory of the test dataset. "
             "Expected structure: dataset/subject_XXX/{face,fingerprint}_*.jpg",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory for output figures and tables. Defaults to ./results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation. Defaults to 32.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Image size for preprocessing. Defaults to 224.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader worker processes. Defaults to 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )
    parser.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline model evaluation (faster).",
    )
    parser.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip robustness evaluation (faster).",
    )
    parser.add_argument(
        "--skip_uncertainty",
        action="store_true",
        help="Skip uncertainty calibration evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for computation. Defaults to auto (CUDA if available).",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("UFM-Transformer: Evaluation and Baseline Comparison")
    print("=" * 80)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n[1] Device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(device)}")
        print(f"    CUDA Version: {torch.version.cuda}")

    figures_dir, tables_dir = _setup_directories(args.output_dir)
    print(f"\n[2] Output directories:")
    print(f"    Figures: {figures_dir}")
    print(f"    Tables:  {tables_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------------------------------------------------
    # Load Test Dataset
    # -----------------------------------------------------------------------
    print(f"\n[3] Loading test dataset from: {args.dataset_path}")

    if not os.path.exists(args.dataset_path):
        print(f"    ERROR: Dataset path does not exist: {args.dataset_path}")
        print("    Creating dummy dataset for demonstration...")
        from data_loader import _create_dummy_dataset
        args.dataset_path = "/tmp/ufm_eval_dummy_dataset"
        _create_dummy_dataset(args.dataset_path, num_subjects=20, images_per_modality=5)
        print(f"    Dummy dataset created at: {args.dataset_path}")

    loaders = get_dataloaders(
        root_dir=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        seed=args.seed,
        drop_face_prob=0.0,
        drop_fingerprint_prob=0.0,
        apply_quality_degradation=False,
    )

    test_loader = loaders["test"]
    print(f"    Test subjects: {len(loaders['test_subjects'])}")
    print(f"    Test batches:  {len(test_loader)}")

    # -----------------------------------------------------------------------
    # Load Models
    # -----------------------------------------------------------------------
    print(f"\n[4] Loading models...")

    print("  Loading UFM-Transformer...")
    ufm_model = _load_model(
        args.model_path,
        UFMTransformer,
        device,
        feature_dim=512,
        num_fusion_layers=4,
        num_heads=8,
        dropout=0.1,
        use_uncertainty=True,
    )
    print(f"    Parameters: {sum(p.numel() for p in ufm_model.parameters()):,}")

    baseline_models = {}
    if not args.skip_baselines:
        print("  Loading baseline models...")
        baseline_models = _create_baseline_models(device)
        for name, model in baseline_models.items():
            n_params = sum(p.numel() for p in model.parameters())
            print(f"    {name}: {n_params:,} parameters")

    # -----------------------------------------------------------------------
    # Evaluate All Models
    # -----------------------------------------------------------------------
    print(f"\n[5] Evaluating models...")

    all_metrics: Dict[str, Dict[str, float]] = {}
    all_scores: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    print("  Evaluating UFM-Transformer...")
    ufm_metrics, ufm_gen, ufm_imp, ufm_preds = evaluate_model(ufm_model, test_loader, device)
    all_metrics["UFM-Transformer"] = ufm_metrics
    all_scores["UFM-Transformer"] = (ufm_gen, ufm_imp)
    print(f"    EER: {ufm_metrics['eer']:.2f}%, AUC: {ufm_metrics['auc']:.2f}%")

    if not args.skip_baselines:
        for name, model in baseline_models.items():
            print(f"  Evaluating {name}...")
            metrics, gen_scores, imp_scores, preds = evaluate_model(model, test_loader, device)
            all_metrics[name] = metrics
            all_scores[name] = (gen_scores, imp_scores)
            print(f"    EER: {metrics['eer']:.2f}%, AUC: {metrics['auc']:.2f}%")

    # -----------------------------------------------------------------------
    # Robustness Evaluation
    # -----------------------------------------------------------------------
    robustness_results: Optional[Dict[str, Dict[str, float]]] = None
    if not args.skip_robustness:
        print(f"\n[6] Robustness evaluation...")
        robustness_results = evaluate_robustness(ufm_model, test_loader, device)
        for condition, metrics in robustness_results.items():
            print(f"    {condition:30s} - EER: {metrics['eer']:6.2f}%, AUC: {metrics['auc']:6.2f}%")

    # -----------------------------------------------------------------------
    # Uncertainty Calibration Evaluation
    # -----------------------------------------------------------------------
    calibration_data: Optional[Dict[str, Any]] = None
    if not args.skip_uncertainty:
        print(f"\n[7] Uncertainty calibration evaluation...")
        calibration_data = evaluate_uncertainty_calibration(ufm_model, test_loader, device)
        ece, _ = compute_calibration_error(calibration_data["confidences"], calibration_data["accuracies"])
        print(f"    ECE: {ece:.2f}%")
        if len(calibration_data["correct_uncertainties"]) > 0:
            print(f"    Correct uncertainties:   mean={np.mean(calibration_data['correct_uncertainties']):.4f}")
        if len(calibration_data["incorrect_uncertainties"]) > 0:
            print(f"    Incorrect uncertainties: mean={np.mean(calibration_data['incorrect_uncertainties']):.4f}")

    # -----------------------------------------------------------------------
    # Generate Plots
    # -----------------------------------------------------------------------
    print(f"\n[8] Generating plots...")

    for model_name, (gen_scores, imp_scores) in all_scores.items():
        safe_name = model_name.replace(" ", "_").replace("-", "_")
        plot_roc_curve(gen_scores, imp_scores, os.path.join(figures_dir, f"roc_curve_{safe_name}.pdf"))
    print(f"    ROC curves saved ({len(all_scores)} models)")

    fmr, fnmr, _ = compute_det_curve(ufm_gen, ufm_imp)
    plot_det_curve(fmr, fnmr, os.path.join(figures_dir, "det_curve_ufm.pdf"))
    print(f"    DET curve saved")

    plot_score_distributions(ufm_gen, ufm_imp, os.path.join(figures_dir, "score_distributions_ufm.pdf"))
    print(f"    Score distributions saved")

    if len(ufm_preds) > 0:
        all_labels = np.concatenate([
            np.ones(len(ufm_gen), dtype=int),
            np.zeros(len(ufm_imp), dtype=int),
        ])
        n_preds = min(len(ufm_preds), len(all_labels))
        plot_confusion_matrix(ufm_preds[:n_preds], all_labels[:n_preds],
                              os.path.join(figures_dir, "confusion_matrix_ufm.pdf"))
        print(f"    Confusion matrix saved")

    plot_violin_scores(all_scores, os.path.join(figures_dir, "violin_comparison_all_models.pdf"))
    print(f"    Violin comparison saved")

    if calibration_data is not None:
        plot_reliability_diagram(
            calibration_data["confidences"], calibration_data["accuracies"],
            os.path.join(figures_dir, "reliability_diagram.pdf")
        )
        print(f"    Reliability diagram saved")

        if (len(calibration_data["correct_uncertainties"]) > 0 or
                len(calibration_data["incorrect_uncertainties"]) > 0):
            plot_uncertainty_histogram(
                calibration_data["correct_uncertainties"],
                calibration_data["incorrect_uncertainties"],
                os.path.join(figures_dir, "uncertainty_histogram.pdf")
            )
            print(f"    Uncertainty histogram saved")

    print(f"\n    All figures saved to: {figures_dir}")

    # -----------------------------------------------------------------------
    # Export LaTeX Tables
    # -----------------------------------------------------------------------
    print(f"\n[9] Exporting LaTeX tables...")

    export_metrics_latex(
        all_metrics["UFM-Transformer"],
        os.path.join(tables_dir, "metrics_ufm.tex"),
        model_name="UFM-Transformer",
    )
    print(f"    UFM metrics table saved")

    export_comparison_table_latex(all_metrics, os.path.join(tables_dir, "model_comparison.tex"))
    print(f"    Comparison table saved")

    if robustness_results is not None:
        export_robustness_table_latex(robustness_results, os.path.join(tables_dir, "robustness.tex"))
        print(f"    Robustness table saved")

    print(f"\n    All tables saved to: {tables_dir}")

    # -----------------------------------------------------------------------
    # Print Summary
    # -----------------------------------------------------------------------
    print(f"\n[10] Summary Comparison Table")
    print("=" * 80)

    header = f"{'Model':<25} {'EER':>8} {'TAR@0.1%':>10} {'TAR@1%':>10} {'AUC':>8} {'ECE':>8}"
    print(header)
    print("-" * 80)

    for model_name in all_metrics:
        m = all_metrics[model_name]
        row = (
            f"{model_name:<25}"
            f" {m.get('eer', 0):>7.2f}%"
            f" {m.get('tar_at_0_001', 0):>9.2f}%"
            f" {m.get('tar_at_0_01', 0):>9.2f}%"
            f" {m.get('auc', 0):>7.2f}%"
            f" {m.get('ece', 0):>7.2f}%"
        )
        print(row)

    print("=" * 80)

    print("\nBest Model per Metric:")
    metrics_to_check = ["eer", "tar_at_0_001", "auc", "ece"]
    for metric in metrics_to_check:
        if metric in ("eer", "ece"):
            best_model = min(all_metrics, key=lambda k: all_metrics[k].get(metric, float("inf")))
            best_value = all_metrics[best_model].get(metric, 0)
        else:
            best_model = max(all_metrics, key=lambda k: all_metrics[k].get(metric, 0))
            best_value = all_metrics[best_model].get(metric, 0)
        direction = "lower" if metric in ("eer", "ece") else "higher"
        print(f"  {metric.upper():15s}: {best_model} ({best_value:.2f}%, {direction} is better)")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
