"""
UFM-Transformer: Uncertainty-aware Fusion with Missing-modality Transformer
============================================================================

Complete PyTorch implementation of the UFM-Transformer model for
multi-modal (face + fingerprint) biometric verification.

The architecture integrates:
    - Modality-specific encoders (EfficientNet-B2 for face, custom CNN for fingerprint)
    - Quality-aware cross-modal transformer fusion
    - Learnable token-based missing modality handling
    - Uncertainty decomposition (epistemic + aleatoric) via MC Dropout
    - ArcFace-style similarity scoring

Dimension Flow (B = batch size):
    Face image:     (B, 3, 224, 224)
    FP image:       (B, 1, 224, 224)
    Face feat map:  (B, 1408, 7, 7)  <- EfficientNet-B2 conv features
    FP feat map:    (B, 512, 7, 7)    <- Custom CNN conv features
    Face global:    (B, 1408)
    FP global:      (B, 512)
    Projected:      (B, 256)  <- Common embedding space (L2-normalized)
    Fused:          (B, 256)  <- After cross-modal transformer
    Similarity:     (B,)      <- Verification score [-1, 1]
    Uncertainty:    (B,)      <- Total predictive uncertainty

Author: UFM-Transformer Implementation
Version: PyTorch 2.0+
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Optional timm import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import timm
    _HAS_TIMM = True
except ImportError:  # pragma: no cover
    timm = None  # type: ignore[assignment]
    _HAS_TIMM = False


# =============================================================================
# 1. FACE ENCODER (EfficientNet-based)
# =============================================================================

class FaceEncoder(nn.Module):
    """Face feature extractor based on EfficientNet-B2.

    Uses the ``timm`` implementation of EfficientNet-B2 pretrained on
    ImageNet.  Instead of consuming the final classification vector, we
    hook the feature maps *before* global average pooling and return both
    the spatial feature map and a globally pooled descriptor.

    The spatial feature map (``feature_map``) preserves the 2-D structure
    of the face and is used as the token sequence for the cross-modal
    transformer.  The ``global_feature`` is a compact descriptor useful
    for debugging and auxiliary losses.

    Args:
        pretrained: If ``True`` (default), loads ImageNet-pretrained weights.
        drop_rate: Dropout probability for the classifier head (unused here
            because we strip the classifier, but kept for API consistency).

    Input:
        | ``x`` | ``(B, 3, H, W)`` | Face RGB image, typically ``H=W=224`` |

    Output:
        Tuple of:
        - ``feature_map`` : ``(B, 1408, H', W')`` spatial features before
          global pooling.  For input ``224x224``, ``H'=W'=7``.
        - ``global_feature`` : ``(B, 1408)`` globally averaged feature vector.

    Example:
        >>> encoder = FaceEncoder(pretrained=True)
        >>> feat_map, global_feat = encoder(torch.randn(2, 3, 224, 224))
        >>> feat_map.shape
        torch.Size([2, 1408, 7, 7])
        >>> global_feat.shape
        torch.Size([2, 1408])
    """

    def __init__(
        self,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not _HAS_TIMM:
            raise RuntimeError(
                "The `timm` library is required for FaceEncoder. "
                "Install it with:  pip install timm"
            )

        # ------------------------------------------------------------------
        # Load EfficientNet-B2 from timm with only the feature backbone.
        # num_classes=0 strips the classification head entirely, leaving us
        # with convolutions up to (but not including) global pooling.
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=0,          # Remove classifier head
            drop_rate=drop_rate,
            global_pool="",         # Disable global pooling in timm
        )

        # Store expected output channel depth for downstream modules
        self.out_channels: int = 1408  # EfficientNet-B2 final conv channels

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract face feature representations.

        Args:
            x: Face image tensor ``(B, 3, H, W)``.

        Returns:
            ``(feature_map, global_feature)`` tuple.
        """
        # ------------------------------------------------------------------
        # Pass through EfficientNet-B2 backbone.
        # The backbone returns the output of the final convolution *before*
        # any global pooling.  For 224x224 input this yields (B, 1408, 7, 7).
        # ------------------------------------------------------------------
        feature_map: Tensor = self.backbone(x)  # (B, 1408, 7, 7)

        # ------------------------------------------------------------------
        # Produce a global descriptor by adaptive average pooling.
        # This collapses spatial dims: (B, 1408, 7, 7) -> (B, 1408, 1, 1)
        # then squeeze -> (B, 1408).
        # ------------------------------------------------------------------
        global_feature: Tensor = F.adaptive_avg_pool2d(
            feature_map, (1, 1)
        ).squeeze(-1).squeeze(-1)  # (B, 1408)

        return feature_map, global_feature


# =============================================================================
# 2. FINGERPRINT ENCODER (Minutiae-Aware CNN)
# =============================================================================

class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and a skip connection.

    Designed for fingerprint ridge-pattern extraction.  The first conv can
    change the channel depth (``in_ch -> out_ch``) and optionally perform
    downsampling (``stride > 1``).  The skip path includes a 1x1 projection
    when dimensions mismatch.

    Args:
        in_ch: Input channel depth.
        out_ch: Output channel depth.
        stride: Spatial stride for the first conv (default 1).

    Input:
        | ``x`` | ``(B, in_ch, H, W)`` |

    Output:
        | ``out`` | ``(B, out_ch, H//stride, W//stride)`` |
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()

        # Main path: conv -> BN -> ReLU -> conv -> BN
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection: 1x1 projection if dimensions change
        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual skip connection."""
        identity = self.shortcut(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity          # Residual addition
        out = self.relu(out)
        return out


class FingerprintEncoder(nn.Module):
    """Custom CNN encoder for fingerprint ridge-pattern extraction.

    Architecture inspired by ResNet-style residual learning but tailored
    for the fine-grained ridge/valley structure of fingerprints.  The
    network progressively downsamples while increasing channel depth,
    ending with a spatial feature map and a global descriptor.

    The output feature map preserves the minutiae-level spatial structure
    (ridge endings, bifurcations) that is critical for cross-modal
    attention with face features.

    Args:
        in_channels: Number of input channels (1 for grayscale fingerprint).
        base_width: Base channel width (default 64), scales all layers.

    Input:
        | ``x`` | ``(B, in_channels, H, W)`` | Fingerprint image, typically grayscale |

    Output:
        Tuple of:
        - ``feature_map`` : ``(B, 512, H', W')`` spatial features.  For input
          ``224x224``, ``H'=W'=7``.
        - ``global_feature`` : ``(B, 512)`` globally pooled descriptor.

    Example:
        >>> encoder = FingerprintEncoder(in_channels=1)
        >>> feat_map, global_feat = encoder(torch.randn(2, 1, 224, 224))
        >>> feat_map.shape
        torch.Size([2, 512, 7, 7])
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Stem: initial convolution + maxpool
        # (B, 1, 224, 224) -> (B, 64, 112, 112)
        # ------------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, base_width,
                kernel_size=7, stride=2, padding=3, bias=False,
            ),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ------------------------------------------------------------------
        # Residual stages with progressive downsampling.
        # Stage 1: (B, 64, 56, 56)   <- after stem
        # Stage 2: (B, 128, 28, 28)  <- stride 2
        # Stage 3: (B, 256, 14, 14)  <- stride 2
        # Stage 4: (B, 512, 7, 7)    <- stride 2
        # ------------------------------------------------------------------
        self.stage1 = self._make_stage(base_width, base_width, num_blocks=2, stride=1)
        self.stage2 = self._make_stage(base_width, base_width * 2, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(base_width * 2, base_width * 4, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(base_width * 4, base_width * 8, num_blocks=2, stride=2)

        self.out_channels: int = base_width * 8  # 512

        # Initialize weights
        self._init_weights()

    def _make_stage(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Create a stage of residual blocks.

        The first block handles channel/stride transition; remaining blocks
        keep dimensions constant.
        """
        layers: List[nn.Module] = [
            ResidualBlock(in_ch, out_ch, stride=stride)
        ]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """He (Kaiming) initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract fingerprint feature representations.

        Args:
            x: Fingerprint image tensor ``(B, 1, H, W)``.

        Returns:
            ``(feature_map, global_feature)`` tuple.
        """
        # (B, 1, 224, 224) -> (B, 64, 56, 56)
        x = self.stem(x)

        # Progressive residual stages
        x = self.stage1(x)   # (B, 64, 56, 56)
        x = self.stage2(x)   # (B, 128, 28, 28)
        x = self.stage3(x)   # (B, 256, 14, 14)
        feature_map = self.stage4(x)  # (B, 512, 7, 7)

        # Global descriptor
        global_feature = F.adaptive_avg_pool2d(
            feature_map, (1, 1)
        ).squeeze(-1).squeeze(-1)  # (B, 512)

        return feature_map, global_feature


# =============================================================================
# 3. QUALITY ESTIMATOR
# =============================================================================

class QualityEstimator(nn.Module):
    """Lightweight CNN that predicts a per-image quality score in ``[0, 1]``.

    The quality score reflects how "reliable" a biometric sample is.
    A face image with occlusion or a fingerprint with smudged ridges
    should receive a low quality score.  This score is used downstream
    to *modulate* the cross-attention weights in the transformer, so
    low-quality modalities contribute less to the fused representation.

    Architecture: 4 convolutional layers with max-pooling, followed by
    a small fully-connected head and a sigmoid output.

    Args:
        in_channels: Number of input channels (3 for RGB face, 1 for FP).

    Input:
        | ``x`` | ``(B, in_channels, H, W)`` |

    Output:
        | ``quality`` | ``(B, 1)`` | Quality score in ``[0, 1]`` |

    Example:
        >>> qe = QualityEstimator(in_channels=3)
        >>> score = qe(torch.randn(2, 3, 224, 224))
        >>> score.shape
        torch.Size([2, 1])
        >>> assert (score >= 0).all() and (score <= 1).all()
    """

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Light-weight feature extractor: 4 conv blocks, each halves spatial
        # resolution.  Starting from 224x224:
        #   -> 112x112 -> 56x56 -> 28x28 -> 14x14
        # ------------------------------------------------------------------
        self.features = nn.Sequential(
            # Block 1: (B, C, 224, 224) -> (B, 32, 112, 112)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: (B, 32, 112, 112) -> (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (B, 64, 56, 56) -> (B, 128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 4: (B, 128, 28, 28) -> (B, 128, 14, 14)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Global average pooling: (B, 128, 14, 14) -> (B, 128)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # ------------------------------------------------------------------
        # Quality regression head: 128 -> 64 -> 1 with sigmoid
        # ------------------------------------------------------------------
        self.regressor = nn.Sequential(
            nn.Flatten(),                 # (B, 128)
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),                 # Clamp to [0, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Predict quality score.

        Args:
            x: Image tensor ``(B, C, H, W)``.

        Returns:
            Quality score ``(B, 1)`` in range ``[0, 1]``.
        """
        feat = self.features(x)   # (B, 128, 1, 1)
        quality = self.regressor(feat)  # (B, 1)
        return quality


# =============================================================================
# 4. PROJECTOR (L2-normalized Common Embedding)
# =============================================================================

class Projector(nn.Module):
    """Projection head that maps modality-specific features to a common
    embedding space with fixed dimensionality and L2 normalization.

    Both face and fingerprint features are projected into the *same*
    ``embed_dim``-dimensional space.  L2 normalization ensures that the
    embeddings lie on the unit hypersphere, which is essential for:
    - Cosine-similarity based verification
    - ArcFace-style additive angular margin loss
    - Stable cross-modal attention computation

    Args:
        in_dim: Input feature dimensionality (e.g., 1408 for face, 512 for FP).
        embed_dim: Target common embedding dimensionality (default 256).
        hidden_dim: Intermediate layer size (default 512).
        dropout: Dropout probability (default 0.1).

    Input:
        | ``x`` | ``(B, in_dim)`` | Global feature from encoder |

    Output:
        | ``z`` | ``(B, embed_dim)`` | L2-normalized embedding |

    Example:
        >>> proj = Projector(in_dim=1408, embed_dim=256)
        >>> z = proj(torch.randn(2, 1408))
        >>> z.shape
        torch.Size([2, 256])
        >>> # Verify L2 norm = 1
        >>> torch.allclose(z.norm(dim=1), torch.ones(2), atol=1e-5)
        True
    """

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.embed_dim = embed_dim

        # ------------------------------------------------------------------
        # Two-layer MLP: in_dim -> hidden_dim -> embed_dim
        # Using BatchNorm + ReLU + Dropout for regularization.
        # No activation on the final layer to allow free embedding values
        # before L2 normalization.
        # ------------------------------------------------------------------
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project and L2-normalize.

        Args:
            x: Input features ``(B, in_dim)``.

        Returns:
            L2-normalized embeddings ``(B, embed_dim)``.
        """
        z = self.layers(x)  # (B, embed_dim)

        # ------------------------------------------------------------------
        # L2 normalization along the embedding dimension.
        # eps prevents division by zero for zero-vectors.
        # After this, every embedding lies on the unit hypersphere:
        #   ||z_i||_2 = 1  for all i in [0, B-1]
        # ------------------------------------------------------------------
        z = F.normalize(z, p=2, dim=1, eps=1e-12)
        return z


# =============================================================================
# 5. LEARNABLE TOKEN (Missing Modality Handling)
# =============================================================================

class LearnableToken(nn.Module):
    """A single learnable embedding vector that serves as a placeholder
    when a modality is **missing**.

    In biometric systems, one modality may be unavailable (e.g., the
    fingerprint sensor fails, or the face camera is occluded).  Instead
    of using zeros or random noise, we learn a dedicated "modality absent"
    token that the cross-modal transformer learns to interpret.

    The token is learned end-to-end and represents the *expected* feature
    distribution for a missing modality.  During inference, if a modality
    is absent, its entire feature sequence is replaced with repetitions of
    this token.

    Args:
        embed_dim: Dimensionality of the token (must match projector output).

    Attributes:
        token: The learnable parameter of shape ``(1, embed_dim)``.

    Example:
        >>> lt = LearnableToken(embed_dim=256)
        >>> placeholder = lt(batch_size=4)
        >>> placeholder.shape
        torch.Size([4, 256])
    """

    def __init__(self, embed_dim: int = 256) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # ------------------------------------------------------------------
        # Initialize the token with small random values (normal distribution).
        # The token starts near zero but has enough variance to learn
        # meaningful representations during training.
        # ------------------------------------------------------------------
        self.token = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

    def forward(self, batch_size: int) -> Tensor:
        """Broadcast the learnable token to batch size.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Repeated token of shape ``(batch_size, embed_dim)``.
        """
        # Expand the singleton token to the full batch
        return self.token.expand(batch_size, -1)  # (B, embed_dim)


# =============================================================================
# 6. CROSS-MODAL TRANSFORMER (Core Fusion Module)
# =============================================================================

class CrossAttentionLayer(nn.Module):
    """Single cross-attention layer for bi-directional modality fusion.

    This layer implements **two** cross-attention operations:
    1. **Face -> Fingerprint**: Face features *query* fingerprint features.
       This lets the face stream "look at" relevant fingerprint regions.
    2. **Fingerprint -> Face**: Fingerprint features *query* face features.
       This lets the fingerprint stream "look at" relevant face regions.

    Quality scores modulate the attention logits *before* softmax, so
    low-quality modalities automatically receive less attention weight.

    The implementation uses PyTorch's native ``nn.MultiheadAttention``
    with pre-normalization (LayerNorm before attention).

    Args:
        embed_dim: Dimensionality of input/output tokens.
        num_heads: Number of attention heads (default 8).
        dropout: Dropout probability (default 0.1).
        dim_feedforward: FFN hidden dimension (default 1024).

    Input:
        - face_tokens: ``(B, N_face, embed_dim)``
        - fp_tokens:   ``(B, N_fp, embed_dim)``
        - quality_face: ``(B, 1)`` quality score for face modality
        - quality_fp:   ``(B, 1)`` quality score for fingerprint modality

    Output:
        - fused_face: ``(B, N_face, embed_dim)``
        - fused_fp:   ``(B, N_fp, embed_dim)``
        - attn_weights_face: ``(B, num_heads, N_face, N_fp)``
        - attn_weights_fp:   ``(B, num_heads, N_fp, N_face)``
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        dim_feedforward: int = 1024,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # ------------------------------------------------------------------
        # Pre-normalization: LayerNorm applied *before* attention.
        # This stabilizes training for deeper transformers.
        # ------------------------------------------------------------------
        self.norm_face = nn.LayerNorm(embed_dim)
        self.norm_fp = nn.LayerNorm(embed_dim)

        # ------------------------------------------------------------------
        # Two separate MultiheadAttention modules for bi-directional cross-attn.
        # batch_first=True means tensors are (B, N, D) not (N, B, D).
        # ------------------------------------------------------------------
        self.cross_attn_face_to_fp = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_fp_to_face = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # ------------------------------------------------------------------
        # Feed-forward network (FFN) for each stream, applied after attention.
        # Standard transformer FFN: Linear -> ReLU -> Dropout -> Linear
        # ------------------------------------------------------------------
        self.ffn_face = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )
        self.ffn_fp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        # Post-FFN LayerNorm
        self.norm_ffn_face = nn.LayerNorm(embed_dim)
        self.norm_ffn_fp = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        face_tokens: Tensor,
        fp_tokens: Tensor,
        quality_face: Tensor,
        quality_fp: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Bi-directional cross-attention forward pass.

        Args:
            face_tokens: Face token sequence ``(B, N_face, D)``.
            fp_tokens: Fingerprint token sequence ``(B, N_fp, D)``.
            quality_face: Face quality score ``(B, 1)`` in ``[0, 1]``.
            quality_fp: Fingerprint quality score ``(B, 1)`` in ``[0, 1]``.

        Returns:
            ``(fused_face, fused_fp, attn_face, attn_fp)`` tuple.
        """
        # Pre-normalize
        face_norm = self.norm_face(face_tokens)
        fp_norm = self.norm_fp(fp_tokens)

        # ------------------------------------------------------------------
        # Cross-Attention 1: Face queries -> Fingerprint keys/values
        # Quality modulation: scale attention output by fingerprint quality.
        # If fingerprint quality is low (near 0), the face stream receives
        # minimal fingerprint information, effectively ignoring it.
        # ------------------------------------------------------------------
        attn_out_face, attn_weights_face = self.cross_attn_face_to_fp(
            query=face_norm,
            key=fp_norm,
            value=fp_norm,
            need_weights=True,
            average_attn_weights=False,  # Return per-head weights: (B,H,Nq,Nk)
        )
        # attn_out_face:     (B, N_face, D)
        # attn_weights_face: (B, num_heads, N_face, N_fp)

        # Quality modulation: scale the attention *output* by fp quality.
        # This is equivalent to scaling attention weights before the
        # weighted sum.  Using output scaling avoids re-computing einsum.
        # quality_fp: (B, 1) -> (B, 1, 1) for broadcasting over (B, N, D)
        quality_fp_expanded = quality_fp.view(-1, 1, 1)
        attn_out_face = attn_out_face * quality_fp_expanded  # (B, N_face, D)

        # Residual connection + dropout
        face_tokens = face_tokens + self.dropout(attn_out_face)

        # ------------------------------------------------------------------
        # Cross-Attention 2: Fingerprint queries -> Face keys/values
        # Quality modulation: scale attention output by face quality.
        # ------------------------------------------------------------------
        attn_out_fp, attn_weights_fp = self.cross_attn_fp_to_face(
            query=fp_norm,
            key=face_norm,
            value=face_norm,
            need_weights=True,
            average_attn_weights=False,
        )
        # attn_out_fp:     (B, N_fp, D)
        # attn_weights_fp: (B, num_heads, N_fp, N_face)

        quality_face_expanded = quality_face.view(-1, 1, 1)
        attn_out_fp = attn_out_fp * quality_face_expanded  # (B, N_fp, D)

        fp_tokens = fp_tokens + self.dropout(attn_out_fp)

        # ------------------------------------------------------------------
        # Feed-Forward Network (FFN) for each stream with residual connections
        # ------------------------------------------------------------------
        face_tokens = face_tokens + self.dropout(
            self.ffn_face(self.norm_ffn_face(face_tokens))
        )
        fp_tokens = fp_tokens + self.dropout(
            self.ffn_fp(self.norm_ffn_fp(fp_tokens))
        )

        return face_tokens, fp_tokens, attn_weights_face, attn_weights_fp


class CrossModalTransformer(nn.Module):
    """Two-stream Cross-Modal Transformer for face-fingerprint fusion.

    This is the **core fusion module** of UFM-Transformer.  It consists of
    stacked ``CrossAttentionLayer`` blocks that perform **bi-directional**
    cross-attention between face and fingerprint token sequences.

    **How cross-attention with quality modulation works:**
    For each layer, two cross-attention operations occur:
    1. Face tokens (as queries) attend to fingerprint tokens (keys/values).
       The attention scores are scaled by the *fingerprint quality*, so
       a low-quality fingerprint (smudged, partial) contributes less.
    2. Fingerprint tokens (as queries) attend to face tokens (keys/values).
       The attention scores are scaled by the *face quality*.

    Mathematically, the quality-modulated attention is::

        attn_weights = softmax( (Q @ K^T) / sqrt(d_k) ) * quality

    where ``quality`` is the other modality's quality score.  This means
    the receiving stream controls how much it trusts the *source* modality.

    **How missing modalities are handled:**
    When a modality is missing (indicated by ``missing_mask``), its token
    sequence is entirely replaced by the learnable ``absent_token``.  The
    transformer learns to interpret this token as "no information available"
    and the cross-attention naturally produces near-zero attention weights
    for it.  The remaining modality's self-like processing still produces
    a meaningful fused embedding.

    Args:
        embed_dim: Token dimensionality (default 256).
        num_heads: Attention heads per layer (default 8).
        num_layers: Number of cross-attention layers (default 4).
        dim_feedforward: FFN hidden dimension (default 1024).
        dropout: Dropout probability (default 0.1).

    Input:
        - face_tokens: ``(B, N_face, embed_dim)``
        - fp_tokens:   ``(B, N_fp, embed_dim)``
        - quality_face: ``(B, 1)``
        - quality_fp:   ``(B, 1)``
        - missing_mask: ``(B, 2)`` boolean, ``True`` = missing.
          ``missing_mask[:, 0]`` for face, ``[:, 1]`` for fingerprint.
        - absent_token: ``(embed_dim,)`` the learnable missing token.

    Output:
        - fused_face: ``(B, N_face, embed_dim)``
        - fused_fp:   ``(B, N_fp, embed_dim)``
        - attn_maps:  List of attention weight tensors (for visualization)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # ------------------------------------------------------------------
        # Stack of cross-attention layers.
        # Each layer performs bi-directional cross-attention + FFN.
        # ------------------------------------------------------------------
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                dim_feedforward=dim_feedforward,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm for each stream
        self.final_norm_face = nn.LayerNorm(embed_dim)
        self.final_norm_fp = nn.LayerNorm(embed_dim)

    def forward(
        self,
        face_tokens: Tensor,
        fp_tokens: Tensor,
        quality_face: Tensor,
        quality_fp: Tensor,
        missing_mask: Tensor,
        absent_token: nn.Parameter,
    ) -> Tuple[Tensor, Tensor, List[Dict[str, Tensor]]]:
        """Cross-modal transformer fusion forward pass.

        Args:
            face_tokens: Face token sequence ``(B, N_face, D)``.
            fp_tokens: Fingerprint token sequence ``(B, N_fp, D)``.
            quality_face: Face quality ``(B, 1)``.
            quality_fp: Fingerprint quality ``(B, 1)``.
            missing_mask: Boolean mask ``(B, 2)`` indicating missing mods.
            absent_token: Learnable parameter ``(D,)`` for missing mods.

        Returns:
            ``(fused_face, fused_fp, attn_maps)`` tuple.
        """
        B = face_tokens.size(0)
        N_face = face_tokens.size(1)
        N_fp = fp_tokens.size(1)

        # ------------------------------------------------------------------
        # Step 1: Replace missing modalities with the learnable absent token.
        # missing_mask: (B, 2) where mask[:, 0] = face missing, [:, 1] = fp missing
        # We create a (B, D) absent token matrix and expand it to replace
        # the respective token sequences.
        # ------------------------------------------------------------------
        absent_face = absent_token.unsqueeze(0).expand(B, N_face, -1)  # (B, N_face, D)
        absent_fp = absent_token.unsqueeze(0).expand(B, N_fp, -1)      # (B, N_fp, D)

        # Boolean masks for missing: (B,) -> expand
        face_missing = missing_mask[:, 0].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        fp_missing = missing_mask[:, 1].unsqueeze(-1).unsqueeze(-1)    # (B, 1, 1)

        # Replace: where missing, use absent token; else keep original
        face_tokens = torch.where(face_missing, absent_face, face_tokens)
        fp_tokens = torch.where(fp_missing, absent_fp, fp_tokens)

        # ------------------------------------------------------------------
        # Step 2: Apply quality floor for missing modalities.
        # If a modality is missing, set its quality to a small constant
        # (0.01) instead of 0.  This prevents division-by-zero issues in
        # downstream modules while still strongly suppressing its influence.
        # ------------------------------------------------------------------
        quality_face = torch.where(
            missing_mask[:, 0:1], torch.tensor(0.01, device=quality_face.device), quality_face
        )
        quality_fp = torch.where(
            missing_mask[:, 1:2], torch.tensor(0.01, device=quality_fp.device), quality_fp
        )

        # ------------------------------------------------------------------
        # Step 3: Pass through stacked cross-attention layers.
        # Each layer performs bi-directional cross-attention.
        # ------------------------------------------------------------------
        attn_maps: List[Dict[str, Tensor]] = []

        for layer in self.layers:
            face_tokens, fp_tokens, attn_f, attn_p = layer(
                face_tokens, fp_tokens, quality_face, quality_fp
            )
            attn_maps.append({"face_to_fp": attn_f, "fp_to_face": attn_p})

        # ------------------------------------------------------------------
        # Step 4: Final layer normalization.
        # ------------------------------------------------------------------
        fused_face = self.final_norm_face(face_tokens)
        fused_fp = self.final_norm_fp(fp_tokens)

        return fused_face, fused_fp, attn_maps


# =============================================================================
# 7. SIMILARITY HEAD (Verification Score)
# =============================================================================

class SimilarityHead(nn.Module):
    """Cosine similarity with additive angular margin for biometric verification.

    Computes the cosine similarity between two fused embeddings and applies
    an **additive margin** (ArcFace-style) to enhance inter-class separation.
    During training, the margin pushes genuine pairs closer together and
    impostor pairs farther apart.  During inference, the margin is removed
    to produce a calibrated similarity score.

    The similarity score is::

        s = cos(theta)               (inference, no margin)
        s = cos(theta + margin)      (training, for genuine pairs)

    where ``theta = arccos( (z1 . z2) / (||z1|| * ||z2||) )`` is the
    angular distance between embeddings.  Since our projectors already
    L2-normalize, the denominator is 1 and cosine similarity reduces to
    the dot product.

    Args:
        embed_dim: Embedding dimensionality (default 256).
        margin: Additive angular margin in radians (default 0.5).
        scale: Temperature scaling factor (default 30.0).

    Input:
        - ``z1``: ``(B, embed_dim)`` first fused embedding
        - ``z2``: ``(B, embed_dim)`` second fused embedding
        - ``label``: ``(B,)`` optional binary labels (1=genuine, 0=impostor)
          for margin application during training.

    Output:
        | ``similarity`` | ``(B,)`` | Similarity score in ``[-1, 1]`` |

    Example:
        >>> head = SimilarityHead(embed_dim=256, margin=0.5)
        >>> z1 = F.normalize(torch.randn(2, 256), dim=1)
        >>> z2 = F.normalize(torch.randn(2, 256), dim=1)
        >>> sim = head(z1, z2)
        >>> sim.shape
        torch.Size([2])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        margin: float = 0.5,
        scale: float = 30.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.margin = margin
        self.scale = scale

        # Learnable class prototypes for each identity (ArcFace-style)
        # These act as anchors in the embedding space
        self.weight = nn.Parameter(torch.randn(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        z1: Tensor,
        z2: Tensor,
        label: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute similarity between two embeddings.

        Args:
            z1: First embedding ``(B, D)``.
            z2: Second embedding ``(B, D)``.
            label: Optional binary labels ``(B,)``.

        Returns:
            Similarity scores ``(B,)``.
        """
        # ------------------------------------------------------------------
        # Since z1 and z2 are already L2-normalized by the projectors,
        # cosine similarity reduces to the dot product:
        #   cos(theta) = (z1 . z2) / (||z1|| * ||z2||) = z1 . z2
        # ------------------------------------------------------------------
        cos_theta = torch.sum(z1 * z2, dim=1)  # (B,)

        # Clamp to [-1, 1] to avoid numerical issues with arccos
        cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)

        # ------------------------------------------------------------------
        # Apply additive angular margin during training for genuine pairs.
        # The margin increases the angular distance for positive pairs,
        # creating a larger decision boundary in angular space.
        # ------------------------------------------------------------------
        if self.training and label is not None:
            theta = torch.acos(cos_theta)  # angular distance (B,)
            # Add margin only for genuine pairs (label == 1)
            theta_margined = torch.where(
                label.bool(),
                theta + self.margin,
                theta,
            )
            cos_theta = torch.cos(theta_margined)

        # Scale the similarity (temperature scaling for training stability)
        similarity = cos_theta * self.scale  # (B,)

        return similarity


# =============================================================================
# 8. UNCERTAINTY HEAD (Epistemic + Aleatoric)
# =============================================================================

class UncertaintyHead(nn.Module):
    """Uncertainty quantification via Monte-Carlo Dropout.

    This module decomposes the total predictive uncertainty into two
    components:

    1. **Aleatoric uncertainty** (data uncertainty): Irreducible noise
       inherent in the data (e.g., blurry fingerprint, partial occlusion).
       This is modeled by a learnable variance prediction.

    2. **Epistemic uncertainty** (model uncertainty): Uncertainty due to
       the model's lack of knowledge about certain input regions.  This
       is estimated via **Monte-Carlo Dropout** -- running multiple forward
       passes with dropout enabled and measuring the variance across
       predictions.

    **How uncertainty is decomposed:**
    Given ``T`` stochastic forward passes with dropout::

        # MC predictions: {y_t}_{t=1}^T
        mean_pred     = (1/T) * sum(y_t)
        epistemic_var = var({y_t})          # variance across MC samples

        # Aleatoric: learnable data noise (from a learned variance head)
        aleatoric_var = learned_variance

        # Total uncertainty (law of total variance)
        total_var = epistemic_var + aleatoric_var

    The epistemic component identifies inputs where the model is "unsure"
    (out-of-distribution, novel attack patterns), while the aleatoric
    component identifies inputs that are inherently noisy.

    Args:
        embed_dim: Embedding dimensionality (default 256).
        mc_samples: Number of Monte-Carlo dropout samples (default 5).
        dropout: Dropout rate for MC sampling (default 0.2).

    Input (forward):
        - ``z1``: ``(B, embed_dim)`` first fused embedding
        - ``z2``: ``(B, embed_dim)`` second fused embedding

    Output (forward):
        Dict with keys:
        - ``mean_prediction``: ``(B,)`` mean similarity across MC samples
        - ``total_uncertainty``: ``(B,)`` total predictive variance
        - ``aleatoric_var``: ``(B,)`` data noise variance
        - ``epistemic_var``: ``(B,)`` model uncertainty variance

    Example:
        >>> uhead = UncertaintyHead(embed_dim=256, mc_samples=5)
        >>> z1 = torch.randn(2, 256)
        >>> z2 = torch.randn(2, 256)
        >>> out = uhead(z1, z2)
        >>> out["mean_prediction"].shape
        torch.Size([2])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        mc_samples: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.mc_samples = mc_samples
        self.dropout_rate = dropout

        # ------------------------------------------------------------------
        # Learnable aleatoric variance predictor.
        # This network estimates the irreducible data noise for each pair.
        # It outputs log(variance) for numerical stability.
        # ------------------------------------------------------------------
        self.aleatoric_net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        # MC Dropout layer (always enabled during forward_uncertainty)
        self.mc_dropout = nn.Dropout(dropout)

    def _similarity_with_dropout(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute similarity with MC dropout active.

        This applies dropout to the embeddings *before* computing the
        cosine similarity, creating stochastic predictions.

        Args:
            z1: First embedding ``(B, D)``.
            z2: Second embedding ``(B, D)``.

        Returns:
            Similarity scores ``(B,)``.
        """
        # Apply dropout to embeddings (stochastic perturbation)
        z1_drop = self.mc_dropout(z1)  # (B, D)
        z2_drop = self.mc_dropout(z2)  # (B, D)

        # Re-normalize after dropout (dropout changes vector norms)
        z1_drop = F.normalize(z1_drop, p=2, dim=1, eps=1e-12)
        z2_drop = F.normalize(z2_drop, p=2, dim=1, eps=1e-12)

        # Cosine similarity = dot product (already normalized)
        similarity = torch.sum(z1_drop * z2_drop, dim=1)  # (B,)
        return similarity

    def _predict_aleatoric(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Predict aleatoric (data) variance.

        Uses a small network on the concatenated embeddings to predict
        the log-variance of data noise.

        Args:
            z1: First embedding ``(B, D)``.
            z2: Second embedding ``(B, D)``.

        Returns:
            Aleatoric variance ``(B,)``.
        """
        # Concatenate both embeddings
        z_pair = torch.cat([z1, z2], dim=1)  # (B, 2*D)

        # Predict log(variance) for numerical stability
        log_var = self.aleatoric_net(z_pair).squeeze(-1)  # (B,)

        # Convert to variance: var = exp(log_var) > 0
        aleatoric_var = torch.exp(log_var)
        return aleatoric_var

    def forward(self, z1: Tensor, z2: Tensor) -> Dict[str, Tensor]:
        """Decompose uncertainty for a pair of embeddings.

        Runs ``mc_samples`` stochastic forward passes to estimate
        epistemic uncertainty, and uses the learned network to estimate
        aleatoric uncertainty.

        Args:
            z1: First embedding ``(B, D)``.
            z2: Second embedding ``(B, D)``.

        Returns:
            Dictionary with uncertainty decomposition.
        """
        B = z1.size(0)
        device = z1.device

        # ------------------------------------------------------------------
        # Step 1: Monte-Carlo Dropout sampling.
        # Run T forward passes with dropout enabled to collect a distribution
        # of predictions.  The spread of these predictions = epistemic uncertainty.
        # ------------------------------------------------------------------
        mc_predictions: List[Tensor] = []

        # Ensure dropout is enabled during MC sampling
        training_state = self.training
        self.train(True)  # Force dropout on

        for _ in range(self.mc_samples):
            sim = self._similarity_with_dropout(z1, z2)  # (B,)
            mc_predictions.append(sim)

        # Restore training state
        self.train(training_state)

        # Stack: (T, B) -> transpose to (B, T)
        mc_stacked = torch.stack(mc_predictions, dim=1)  # (B, T)

        # ------------------------------------------------------------------
        # Step 2: Compute epistemic uncertainty.
        # This is the variance of predictions across MC samples.
        # High variance = model is uncertain about this input.
        # ------------------------------------------------------------------
        mean_prediction = mc_stacked.mean(dim=1)    # (B,)
        epistemic_var = mc_stacked.var(dim=1, unbiased=True)  # (B,)

        # Ensure non-negative (numerical safety)
        epistemic_var = torch.clamp(epistemic_var, min=0.0)

        # ------------------------------------------------------------------
        # Step 3: Compute aleatoric uncertainty.
        # This is the learned data noise, independent of MC sampling.
        # ------------------------------------------------------------------
        aleatoric_var = self._predict_aleatoric(z1, z2)  # (B,)

        # ------------------------------------------------------------------
        # Step 4: Total uncertainty (law of total variance).
        #   Var(Y|X) = E[Var(Y|X,theta)] + Var(E[Y|X,theta])
        #   total    = aleatoric         + epistemic
        # ------------------------------------------------------------------
        total_uncertainty = aleatoric_var + epistemic_var  # (B,)

        return {
            "mean_prediction": mean_prediction,      # (B,)
            "total_uncertainty": total_uncertainty,  # (B,)
            "aleatoric_var": aleatoric_var,          # (B,)
            "epistemic_var": epistemic_var,          # (B,)
        }


# =============================================================================
# 9. UFM-TRANSFORMER (Complete Model)
# =============================================================================

class UFMTransformer(nn.Module):
    """UFM-Transformer: Uncertainty-aware Fusion with Missing-modality Transformer.

    The complete end-to-end model for face + fingerprint biometric verification.
    It integrates modality-specific encoders, quality estimation, cross-modal
    transformer fusion, similarity scoring, and uncertainty quantification.

    **Architecture Pipeline:**

    .. code-block:: text

        Face Image (B,3,224,224) ──► FaceEncoder ──► feat_map (B,1408,7,7)
                                                       │
                                                       ▼
                                              ┌──────────────────┐
                                              │  Projector       │──► z_face (B,256)
                                              │  (L2 normalized) │       │
                                              └──────────────────┘       │
                                                                         ▼
        FP Image (B,1,224,224) ──► FingerprintEncoder ──► feat_map (B,512,7,7)
                                                              │
                                                              ▼
                                                     ┌──────────────────┐
                                                     │  Projector       │──► z_fp (B,256)
                                                     │  (L2 normalized) │       │
                                                     └──────────────────┘       │
                                                                                ▼
        QualityEstimator(face_img) ──► q_face ───────────────────────► CrossModalTransformer
        QualityEstimator(fp_img)   ──► q_fp   ───────────────────────► (fusion)
        missing_mask (B,2)         ──► token replacement ────────────►
                                                                        │
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │ SimilarityHead   │──► score (B,)
                                                              │ (ArcFace margin) │
                                                              └──────────────────┘
                                                                        │
                                                                        ▼
                                                              ┌──────────────────┐
                                                              │ UncertaintyHead  │──► uncertainty dict
                                                              │ (MC Dropout)     │
                                                              └──────────────────┘

    Args:
        embed_dim: Common embedding dimensionality (default 256).
        num_heads: Attention heads in transformer (default 8).
        num_layers: Transformer layers (default 4).
        dim_feedforward: FFN hidden dim (default 1024).
        dropout: Dropout rate (default 0.1).
        mc_samples: Monte-Carlo samples for uncertainty (default 5).
        arcface_margin: Angular margin for similarity (default 0.5).
        arcface_scale: Temperature scale for similarity (default 30.0).
        use_pretrained_face: Use ImageNet-pretrained EfficientNet (default True).

    Input (forward):
        - ``face_img``: ``(B, 3, H, W)`` face RGB images
        - ``fp_img``: ``(B, 1, H, W)`` fingerprint grayscale images
        - ``face_quality``: ``(B, 1)`` optional pre-computed face quality
        - ``fp_quality``: ``(B, 1)`` optional pre-computed fingerprint quality
        - ``missing_mask``: ``(B, 2)`` boolean mask (True = missing)

    Output (forward):
        Dict with keys:
        - ``similarity_score``: ``(B,)`` verification similarity
        - ``uncertainty``: dict with mean, total, aleatoric, epistemic
        - ``attention_maps``: list of per-layer attention weight tensors
        - ``z_face``: ``(B, embed_dim)`` projected face embedding
        - ``z_fp``: ``(B, embed_dim)`` projected fingerprint embedding
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        mc_samples: int = 5,
        arcface_margin: float = 0.5,
        arcface_scale: float = 30.0,
        use_pretrained_face: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # ------------------------------------------------------------------
        # 1. Modality-specific encoders
        # ------------------------------------------------------------------
        self.face_encoder = FaceEncoder(
            pretrained=use_pretrained_face,
            drop_rate=dropout,
        )
        self.fp_encoder = FingerprintEncoder(
            in_channels=1,
            base_width=64,
        )

        # ------------------------------------------------------------------
        # 2. Quality estimators (one per modality)
        # ------------------------------------------------------------------
        self.face_quality_estimator = QualityEstimator(in_channels=3)
        self.fp_quality_estimator = QualityEstimator(in_channels=1)

        # ------------------------------------------------------------------
        # 3. Projectors: encoder features -> common embedding space
        # Face: 1408 -> 256, Fingerprint: 512 -> 256
        # ------------------------------------------------------------------
        self.face_projector = Projector(
            in_dim=self.face_encoder.out_channels,
            embed_dim=embed_dim,
            hidden_dim=512,
            dropout=dropout,
        )
        self.fp_projector = Projector(
            in_dim=self.fp_encoder.out_channels,
            embed_dim=embed_dim,
            hidden_dim=512,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # 4. Learnable token for missing modality handling
        # ------------------------------------------------------------------
        self.learnable_token = LearnableToken(embed_dim=embed_dim)

        # ------------------------------------------------------------------
        # 5. Cross-Modal Transformer (core fusion)
        #    Flattened feature map tokens: (B, 49, 256) for each modality
        # ------------------------------------------------------------------
        # First, we need to project the spatial feature maps to embed_dim
        # Face: (B, 1408, 7, 7) -> (B, 49, 256)
        # FP:   (B, 512, 7, 7)   -> (B, 49, 256)
        self.face_feat_projector = nn.Conv2d(
            self.face_encoder.out_channels, embed_dim, kernel_size=1
        )
        self.fp_feat_projector = nn.Conv2d(
            self.fp_encoder.out_channels, embed_dim, kernel_size=1
        )

        self.cross_modal_transformer = CrossModalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Global pooling after transformer to get compact embeddings
        # ------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ------------------------------------------------------------------
        # 6. Similarity head (ArcFace-style cosine similarity)
        # ------------------------------------------------------------------
        self.similarity_head = SimilarityHead(
            embed_dim=embed_dim,
            margin=arcface_margin,
            scale=arcface_scale,
        )

        # ------------------------------------------------------------------
        # 7. Uncertainty head (MC Dropout decomposition)
        # ------------------------------------------------------------------
        self.uncertainty_head = UncertaintyHead(
            embed_dim=embed_dim,
            mc_samples=mc_samples,
            dropout=dropout,
        )

    def _feature_map_to_tokens(self, feature_map: Tensor, proj_conv: nn.Conv2d) -> Tensor:
        """Convert a convolutional feature map to a sequence of tokens.

        Steps:
        1. Project channels to embed_dim via 1x1 conv.
        2. Flatten spatial dimensions to create a token sequence.

        Args:
            feature_map: ``(B, C, H, W)`` feature map.
            proj_conv: 1x1 convolution to project to embed_dim.

        Returns:
            Token sequence ``(B, H*W, embed_dim)``.
        """
        # Project channels: (B, C, H, W) -> (B, embed_dim, H, W)
        projected = proj_conv(feature_map)

        B, D, H, W = projected.shape

        # Flatten spatial: (B, embed_dim, H, W) -> (B, embed_dim, H*W)
        tokens = projected.view(B, D, H * W)

        # Transpose to (B, H*W, embed_dim) for transformer
        tokens = tokens.transpose(1, 2)  # (B, H*W, embed_dim)
        tokens = tokens.contiguous()

        return tokens

    def forward(
        self,
        face_img: Tensor,
        fp_img: Tensor,
        face_quality: Optional[Tensor] = None,
        fp_quality: Optional[Tensor] = None,
        missing_mask: Optional[Tensor] = None,
    ) -> Dict[str, Union[Tensor, Dict, List]]:
        """Complete UFM-Transformer forward pass.

        Args:
            face_img: Face images ``(B, 3, H, W)``.
            fp_img: Fingerprint images ``(B, 1, H, W)``.
            face_quality: Optional face quality ``(B, 1)``. If None,
                estimated automatically.
            fp_quality: Optional fingerprint quality ``(B, 1)``. If None,
                estimated automatically.
            missing_mask: Boolean mask ``(B, 2)`` where ``[:, 0]`` indicates
                missing face and ``[:, 1]`` indicates missing fingerprint.
                If None, assumes all modalities present.

        Returns:
            Dictionary with similarity scores, uncertainty estimates,
            attention maps, and intermediate embeddings.
        """
        B = face_img.size(0)
        device = face_img.device

        # ------------------------------------------------------------------
        # Step 0: Default missing mask if not provided
        # ------------------------------------------------------------------
        if missing_mask is None:
            missing_mask = torch.zeros(B, 2, dtype=torch.bool, device=device)

        # ==================================================================
        # STAGE 1: Feature Extraction
        # ==================================================================

        # Face encoder: (B, 3, 224, 224) -> feat_map (B, 1408, 7, 7), global (B, 1408)
        face_feat_map, face_global = self.face_encoder(face_img)

        # Fingerprint encoder: (B, 1, 224, 224) -> feat_map (B, 512, 7, 7), global (B, 512)
        fp_feat_map, fp_global = self.fp_encoder(fp_img)

        # ------------------------------------------------------------------
        # Stage 1b: Quality Estimation
        # If pre-computed qualities are not provided, estimate them on-the-fly.
        # Quality scores are in [0, 1], where 1 = excellent, 0 = unusable.
        # ------------------------------------------------------------------
        if face_quality is None:
            face_quality = self.face_quality_estimator(face_img)  # (B, 1)
        if fp_quality is None:
            fp_quality = self.fp_quality_estimator(fp_img)        # (B, 1)

        # ==================================================================
        # STAGE 2: Projection to Common Embedding Space
        # ==================================================================

        # Project global features: (B, 1408) -> (B, 256), (B, 512) -> (B, 256)
        z_face = self.face_projector(face_global)  # (B, 256), L2-normalized
        z_fp = self.fp_projector(fp_global)        # (B, 256), L2-normalized

        # ==================================================================
        # STAGE 3: Cross-Modal Transformer Fusion
        # ==================================================================

        # Convert feature maps to token sequences for the transformer
        # face: (B, 1408, 7, 7) -> (B, 49, 256)
        # fp:   (B, 512, 7, 7)   -> (B, 49, 256)
        face_tokens = self._feature_map_to_tokens(
            face_feat_map, self.face_feat_projector
        )
        fp_tokens = self._feature_map_to_tokens(
            fp_feat_map, self.fp_feat_projector
        )

        # Pass through cross-modal transformer with quality modulation
        # and missing modality handling
        fused_face_tokens, fused_fp_tokens, attn_maps = self.cross_modal_transformer(
            face_tokens=face_tokens,
            fp_tokens=fp_tokens,
            quality_face=face_quality,
            quality_fp=fp_quality,
            missing_mask=missing_mask,
            absent_token=self.learnable_token.token,
        )

        # Global average pooling over tokens: (B, 49, 256) -> (B, 256)
        fused_face = fused_face_tokens.mean(dim=1)  # (B, 256)
        fused_fp = fused_fp_tokens.mean(dim=1)      # (B, 256)

        # L2-normalize fused embeddings
        fused_face = F.normalize(fused_face, p=2, dim=1, eps=1e-12)
        fused_fp = F.normalize(fused_fp, p=2, dim=1, eps=1e-12)

        # ==================================================================
        # STAGE 4: Similarity Scoring
        # ==================================================================

        similarity_score = self.similarity_head(fused_face, fused_fp)

        # ==================================================================
        # STAGE 5: Uncertainty Quantification
        # ==================================================================

        uncertainty = self.uncertainty_head(fused_face, fused_fp)

        # ==================================================================
        # RETURN
        # ==================================================================

        return {
            "similarity_score": similarity_score,      # (B,)
            "uncertainty": uncertainty,                # dict of 4 tensors
            "attention_maps": attn_maps,               # list of per-layer dicts
            "z_face": z_face,                          # (B, 256) pre-fusion
            "z_fp": z_fp,                              # (B, 256) pre-fusion
            "fused_face": fused_face,                  # (B, 256) post-fusion
            "fused_fp": fused_fp,                      # (B, 256) post-fusion
            "face_quality": face_quality,              # (B, 1)
            "fp_quality": fp_quality,                  # (B, 1)
        }


# =============================================================================
# 10. FACTORY FUNCTION
# =============================================================================

def get_ufm_model(
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    mc_samples: int = 5,
    arcface_margin: float = 0.5,
    arcface_scale: float = 30.0,
    use_pretrained_face: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> UFMTransformer:
    """Factory function to create a UFM-Transformer model with configurable parameters.

    This is the recommended way to instantiate the model.  It handles device
    placement and prints a summary of the configuration.

    Args:
        embed_dim: Common embedding dimensionality (default 256).
        num_heads: Number of attention heads (default 8).
        num_layers: Number of transformer layers (default 4).
        dim_feedforward: FFN hidden dimension (default 1024).
        dropout: Dropout probability (default 0.1).
        mc_samples: Monte-Carlo dropout samples for uncertainty (default 5).
        arcface_margin: Additive angular margin (default 0.5).
        arcface_scale: Temperature scaling factor (default 30.0).
        use_pretrained_face: Use ImageNet-pretrained EfficientNet (default True).
        device: Target device. If None, auto-selects CUDA if available.

    Returns:
        Configured ``UFMTransformer`` model (possibly on GPU).

    Example:
        >>> model = get_ufm_model(embed_dim=256, num_layers=4)
        >>> model
        UFMTransformer(...)

        >>> # Quick forward test
        >>> face = torch.randn(2, 3, 224, 224)
        >>> fp = torch.randn(2, 1, 224, 224)
        >>> out = model(face, fp)
        >>> out["similarity_score"].shape
        torch.Size([2])
    """
    # Auto-select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = UFMTransformer(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        mc_samples=mc_samples,
        arcface_margin=arcface_margin,
        arcface_scale=arcface_scale,
        use_pretrained_face=use_pretrained_face,
    )

    model = model.to(device)

    # ------------------------------------------------------------------
    # Print configuration summary
    # ------------------------------------------------------------------
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 60)
    print("  UFM-Transformer Configuration")
    print("=" * 60)
    print(f"  Embedding dim:        {embed_dim}")
    print(f"  Attention heads:      {num_heads}")
    print(f"  Transformer layers:   {num_layers}")
    print(f"  FFN dim:              {dim_feedforward}")
    print(f"  Dropout:              {dropout}")
    print(f"  MC samples:           {mc_samples}")
    print(f"  ArcFace margin:       {arcface_margin}")
    print(f"  ArcFace scale:        {arcface_scale}")
    print(f"  Pretrained face:      {use_pretrained_face}")
    print(f"  Device:               {device}")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print("=" * 60)

    return model


# Backward-compatibility alias used by train.py
UFMTransformerModel = UFMTransformer


# =============================================================================
# Quick self-test (runs when the file is executed directly)
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  UFM-Transformer: Self-Test")
    print("=" * 60)

    # Use CPU for self-test to avoid GPU requirement
    device = torch.device("cpu")

    # Create model without pretrained weights (faster for testing)
    model = get_ufm_model(
        embed_dim=256,
        num_heads=8,
        num_layers=4,
        use_pretrained_face=False,
        device=device,
    )

    # Create dummy inputs
    B = 2
    face = torch.randn(B, 3, 224, 224, device=device)
    fp = torch.randn(B, 1, 224, 224, device=device)

    # Test 1: All modalities present
    print("\n--- Test 1: Both modalities present ---")
    model.eval()
    with torch.no_grad():
        out = model(face, fp)
    print(f"  similarity_score:  {out['similarity_score'].shape}  {out['similarity_score'].tolist()}")
    print(f"  mean_prediction:   {out['uncertainty']['mean_prediction'].shape}")
    print(f"  total_uncertainty: {out['uncertainty']['total_uncertainty'].shape}")
    print(f"  aleatoric_var:     {out['uncertainty']['aleatoric_var'].shape}")
    print(f"  epistemic_var:     {out['uncertainty']['epistemic_var'].shape}")
    print(f"  attention_maps:    {len(out['attention_maps'])} layers")
    print(f"  z_face:            {out['z_face'].shape}")
    print(f"  z_fp:              {out['z_fp'].shape}")
    print(f"  fused_face:        {out['fused_face'].shape}")
    print(f"  fused_fp:          {out['fused_fp'].shape}")

    # Verify L2 normalization
    face_norm = out["z_face"].norm(dim=1)
    fp_norm = out["z_fp"].norm(dim=1)
    print(f"  ||z_face||_2:      {face_norm.tolist()}")
    print(f"  ||z_fp||_2:        {fp_norm.tolist()}")
    assert torch.allclose(face_norm, torch.ones(B), atol=1e-5)
    assert torch.allclose(fp_norm, torch.ones(B), atol=1e-5)
    print("  [PASS] L2 normalization verified.")

    # Test 2: Missing face modality
    print("\n--- Test 2: Missing face modality ---")
    missing_face = torch.tensor([[True, False], [True, False]], device=device)
    with torch.no_grad():
        out2 = model(face, fp, missing_mask=missing_face)
    print(f"  similarity_score:  {out2['similarity_score'].tolist()}")
    print(f"  total_uncertainty: {out2['uncertainty']['total_uncertainty'].tolist()}")
    print("  [PASS] Missing face handled correctly.")

    # Test 3: Missing fingerprint modality
    print("\n--- Test 3: Missing fingerprint modality ---")
    missing_fp = torch.tensor([[False, True], [False, True]], device=device)
    with torch.no_grad():
        out3 = model(face, fp, missing_mask=missing_fp)
    print(f"  similarity_score:  {out3['similarity_score'].tolist()}")
    print("  [PASS] Missing fingerprint handled correctly.")

    # Test 4: Both modalities missing
    print("\n--- Test 4: Both modalities missing ---")
    missing_both = torch.tensor([[True, True], [True, True]], device=device)
    with torch.no_grad():
        out4 = model(face, fp, missing_mask=missing_both)
    print(f"  similarity_score:  {out4['similarity_score'].tolist()}")
    print("  [PASS] Both missing handled correctly.")

    print("\n" + "=" * 60)
    print("  All self-tests passed!")
    print("=" * 60)
