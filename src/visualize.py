"""
Visualization of attention maps and Grad-CAM for multimodal biometrics.

Generates interpretability plots showing which face regions and fingerprint
minutiae contribute to the verification decision. All outputs are saved as
high-resolution PDF (300 DPI) or multi-page PDF reports.

Functions:
    extract_attention_maps:
        Extract cross-attention weights from the UFM-Transformer model.
    visualize_cross_attention:
        Plot attention between face regions and fingerprint patches.
    compute_gradcam_bimodal:
        Grad-CAM for face and fingerprint branches.
    visualize_gradcam_bimodal:
        Overlay Grad-CAM heatmaps on original images.
    generate_explainability_report:
        Generate Grad-CAM and attention visualizations for test identities.
    plot_attention_heatmap:
        Plot cross-modal attention matrix as heatmap.

Classes:
    GradCAMBimodal:
        Computes Grad-CAM heatmaps independently for face and fingerprint
        branches of the bimodal network.

Author: UFM-Transformer Team
"""

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_pdf import PdfPages

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DPI: int = 300  # Resolution for all saved figures
_CMAP_FACE: str = "jet"  # Colour map for face Grad-CAM
_CMAP_FP: str = "jet"  # Colour map for fingerprint Grad-CAM
_CMAP_ATTN: str = "viridis"  # Colour map for attention heatmaps
_FIG_WIDTH: float = 12.0  # Default figure width (inches)
_FIG_HEIGHT: float = 8.0  # Default figure height (inches)


# ---------------------------------------------------------------------------
# Attention map extraction
# ---------------------------------------------------------------------------


def extract_attention_maps(
    model: nn.Module,
    face_img: torch.Tensor,
    fp_img: torch.Tensor,
    layer_name: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """Extract cross-attention weights from the UFM-Transformer.

    Registers forward hooks on ``nn.MultiheadAttention`` layers inside the
    model, runs a single forward pass, and collects the attention weight
    tensors.  Attention tensors have shape
    ``(batch, num_heads, seq_q, seq_k)``.

    Args:
        model: UFM-Transformer model (or any model containing MHA layers).
        face_img: Face image tensor of shape ``(B, C, H, W)``.
        fp_img: Fingerprint image tensor of shape ``(B, C, H, W)``.
        layer_name: If provided, only extract attention from layers whose
            name contains this substring (e.g. ``"cross_attn"``).

    Returns:
        Dictionary mapping layer names to attention weight tensors.
        Each tensor has shape ``(batch, num_heads, seq_q, seq_k)``.

    Example:
        >>> attn = extract_attention_maps(model, face_batch, fp_batch)
        >>> for name, weight in attn.items():
        ...     print(name, weight.shape)
    """
    attention_maps: Dict[str, torch.Tensor] = {}
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(name: str):
        """Create a hook that captures attention weights."""
        def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
            # MultiheadAttention outputs (attn_output, attn_output_weights)
            # in training mode or when need_weights=True
            if isinstance(outputs, tuple) and len(outputs) == 2:
                _, attn_weights = outputs
                if attn_weights is not None:
                    attention_maps[name] = attn_weights.detach().cpu()
        return hook

    # Register hooks on all MultiheadAttention modules
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if layer_name is None or layer_name in name:
                handles.append(module.register_forward_hook(_make_hook(name)))

    if not handles:
        logging.warning("No MultiheadAttention layers found in the model")

    # Forward pass – set need_weights=True via forward pre-hook if possible
    model.eval()
    with torch.no_grad():
        try:
            model(face_img, fp_img)
        except TypeError:
            # Model may expect a single tuple or dict input
            model((face_img, fp_img))

    # Clean up hooks
    for h in handles:
        h.remove()

    logging.info("Extracted attention maps from %d layer(s)", len(attention_maps))
    return attention_maps


# ---------------------------------------------------------------------------
# Cross-attention visualization
# ---------------------------------------------------------------------------


def visualize_cross_attention(
    attention_maps: Dict[str, torch.Tensor],
    face_img: torch.Tensor,
    fp_img: torch.Tensor,
    save_path: Union[str, Path],
    max_heads: int = 8,
    layer_filter: Optional[str] = None,
) -> None:
    """Plot attention between face regions and fingerprint patches.

    For each selected cross-attention layer and each head (up to ``max_heads``),
    creates a grid showing which face patches attend to which fingerprint
    patches. The first sample in the batch is visualised.

    Args:
        attention_maps: Output of :func:`extract_attention_maps`.
        face_img: Face image tensor ``(B, C, H, W)`` – first sample plotted.
        fp_img: Fingerprint image tensor ``(B, C, H, W)`` – first sample plotted.
        save_path: Path for the output PDF figure.
        max_heads: Maximum number of attention heads to plot per layer.
        layer_filter: If set, only layers whose name contains this string are
            visualised.

    Returns:
        ``None`` (figure saved to disk).

    Example:
        >>> attn = extract_attention_maps(model, f_img, fp_img)
        >>> visualize_cross_attention(attn, f_img, fp_img, "cross_attn.pdf")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter layers
    selected_layers = {
        k: v for k, v in attention_maps.items()
        if layer_filter is None or layer_filter in k
    }

    if not selected_layers:
        logging.warning("No attention layers match filter '%s'", layer_filter)
        return

    # Convert first image in batch to displayable numpy
    face_np = _tensor_to_display(face_img[0])  # (H, W, 3)
    fp_np = _tensor_to_display(fp_img[0])  # (H, W, 3)

    n_layers = len(selected_layers)
    with PdfPages(save_path) as pdf:
        for layer_name, attn_tensor in selected_layers.items():
            # attn_tensor: (B, num_heads, seq_q, seq_k)
            b, num_heads, seq_q, seq_k = attn_tensor.shape
            n_plot_heads = min(num_heads, max_heads)
            ncols = min(n_plot_heads, 4)
            nrows = (n_plot_heads + ncols - 1) // ncols

            fig, axes = plt.subplots(
                nrows, ncols,
                figsize=(_FIG_WIDTH, 2.5 * nrows),
                squeeze=False,
            )
            fig.suptitle(
                f"Cross-Attention: {layer_name}\n"
                f"(Q: {seq_q} patches, K: {seq_k} patches, Heads: {num_heads})",
                fontsize=10,
            )

            for h_idx in range(n_plot_heads):
                row, col = divmod(h_idx, ncols)
                ax = axes[row][col]
                # Average attention for this head across the batch
                attn_head = attn_tensor[0, h_idx].numpy()  # (seq_q, seq_k)
                im = ax.imshow(attn_head, cmap=_CMAP_ATTN, aspect="auto")
                ax.set_title(f"Head {h_idx}", fontsize=8)
                ax.set_xlabel("Fingerprint patches", fontsize=7)
                ax.set_ylabel("Face patches", fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Hide unused subplots
            for h_idx in range(n_plot_heads, nrows * ncols):
                row, col = divmod(h_idx, ncols)
                axes[row][col].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=_DPI)
            plt.close(fig)

    logging.info("Cross-attention visualisation saved to %s", save_path)


# ---------------------------------------------------------------------------
# Grad-CAM implementation for bimodal network
# ---------------------------------------------------------------------------


class GradCAMBimodal:
    """Grad-CAM for bimodal (face + fingerprint) networks.

    Computes class-discriminative localization maps independently for each
    modality branch by using gradient flow back-propagated from the target
    class score.

    Args:
        model: The trained UFM-Transformer model.
        face_target_layer: Name of the convolutional feature layer in the
            face branch whose activations are used for Grad-CAM.
        fp_target_layer: Name of the convolutional feature layer in the
            fingerprint branch.

    Example:
        >>> gradcam = GradCAMBimodal(model, "face_cnn.layer4",
        ...                          "fp_cnn.layer4")
        >>> face_hm, fp_hm = gradcam(face_img, fp_img, target_class=1)
    """

    def __init__(
        self,
        model: nn.Module,
        face_target_layer: str,
        fp_target_layer: str,
    ) -> None:
        self.model = model
        self.model.eval()

        self.face_target_layer = face_target_layer
        self.fp_target_layer = fp_target_layer

        # Storage for forward activations and backward gradients
        self._face_activation: Optional[torch.Tensor] = None
        self._face_gradient: Optional[torch.Tensor] = None
        self._fp_activation: Optional[torch.Tensor] = None
        self._fp_gradient: Optional[torch.Tensor] = None

        self._face_handle_f: Optional[Any] = None
        self._face_handle_b: Optional[Any] = None
        self._fp_handle_f: Optional[Any] = None
        self._fp_handle_b: Optional[Any] = None

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layers."""

        def _get_forward_hook(storage_attr: str):
            def hook(module: nn.Module, inp: Any, out: torch.Tensor) -> None:
                setattr(self, storage_attr, out.detach())
            return hook

        def _get_backward_hook(storage_attr: str):
            def hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
                # grad_output is a tuple; take first element
                grad = grad_output[0].detach() if isinstance(grad_output, tuple) else grad_output.detach()
                setattr(self, storage_attr, grad)
            return hook

        # Locate target layers and register hooks
        for name, module in self.model.named_modules():
            if name == self.face_target_layer:
                self._face_handle_f = module.register_forward_hook(
                    _get_forward_hook("_face_activation")
                )
                self._face_handle_b = module.register_full_backward_hook(
                    _get_backward_hook("_face_gradient")
                )
            if name == self.fp_target_layer:
                self._fp_handle_f = module.register_forward_hook(
                    _get_forward_hook("_fp_activation")
                )
                self._fp_handle_b = module.register_full_backward_hook(
                    _get_backward_hook("_fp_gradient")
                )

        if self._face_handle_f is None:
            raise ValueError(f"Face target layer '{self.face_target_layer}' not found")
        if self._fp_handle_f is None:
            raise ValueError(f"FP target layer '{self.fp_target_layer}' not found")

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in (self._face_handle_f, self._face_handle_b,
                       self._fp_handle_f, self._fp_handle_b):
            if handle is not None:
                handle.remove()
        self._face_handle_f = self._face_handle_b = None
        self._fp_handle_f = self._fp_handle_b = None

    def __call__(
        self,
        face_img: torch.Tensor,
        fp_img: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Grad-CAM heatmaps for both modalities.

        Args:
            face_img: Face image ``(B, C, H, W)``.
            fp_img: Fingerprint image ``(B, C, H, W)``.
            target_class: Class index to explain. ``None`` uses the predicted
                class. For biometric verification, ``1`` = genuine, ``0`` =
                impostor.

        Returns:
            Tuple of ``(face_heatmap, fp_heatmap)`` as numpy arrays scaled
            to ``[0, 1]`` with shapes ``(H, W)`` matching the input image
            resolution.
        """
        self._register_hooks()
        self.model.zero_grad()

        # Forward pass
        output = self.model(face_img, fp_img)

        # Handle different output formats
        if isinstance(output, dict):
            logits = output.get("logits", output.get("score", None))
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if logits is None:
            raise ValueError("Could not extract logits from model output")

        # If logits are 1-D (single score), squeeze
        if logits.dim() == 2 and logits.shape[1] == 1:
            score = logits[:, 0]
        else:
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1)[0])
            score = logits[0, target_class]

        # Backward pass
        score.backward()

        # Compute Grad-CAM for face branch
        face_cam = self._compute_cam(self._face_activation, self._face_gradient,
                                     face_img.shape[2:])

        # Compute Grad-CAM for fingerprint branch
        fp_cam = self._compute_cam(self._fp_activation, self._fp_gradient,
                                   fp_img.shape[2:])

        self._remove_hooks()
        return face_cam, fp_cam

    @staticmethod
    def _compute_cam(
        activation: Optional[torch.Tensor],
        gradient: Optional[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        """Compute Grad-CAM from stored activation and gradient.

        Implements the standard Grad-CAM formulation:
        :math:`CAM = ReLU(\\sum_k \\alpha_k A^k)` where
        :math:`\\alpha_k = \\frac{1}{Z} \\sum_{i,j} \\frac{\\partial y}{\\partial A^k_{ij}}`.

        Args:
            activation: Forward activation tensor ``(B, C, H_a, W_a)``.
            gradient: Backward gradient tensor ``(B, C, H_a, W_a)``.
            target_size: ``(H, W)`` to upsample the CAM to.

        Returns:
            Normalised CAM heatmap ``(H, W)`` in ``[0, 1]``.
        """
        if activation is None or gradient is None:
            # Return zero heatmap if hooks did not fire
            return np.zeros(target_size, dtype=np.float32)

        # Take first sample in batch
        activation = activation[0]  # (C, H, W)
        gradient = gradient[0]  # (C, H, W)

        # Global average pooling of gradients per channel -> weights
        weights = torch.mean(gradient, dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * activation, dim=0)  # (H, W)

        # ReLU
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Upsample to target size
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        cam = F.interpolate(
            cam, size=target_size, mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().detach().cpu().numpy()

        return cam

    def release(self) -> None:
        """Release hooks and free stored tensors."""
        self._remove_hooks()
        self._face_activation = self._face_gradient = None
        self._fp_activation = self._fp_gradient = None


def compute_gradcam_bimodal(
    model: nn.Module,
    face_img: torch.Tensor,
    fp_img: torch.Tensor,
    target_class: Optional[int] = None,
    face_layer: str = "face_cnn.layer4",
    fp_layer: str = "fp_cnn.layer4",
) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper for bimodal Grad-CAM computation.

    Computes Grad-CAM independently for face and fingerprint branches,
    producing heatmaps that highlight salient regions for the *target_class*
    decision.

    Args:
        model: Trained UFM-Transformer model.
        face_img: Face image tensor ``(B, C, H, W)``.
        fp_img: Fingerprint image tensor ``(B, C, H, W)``.
        target_class: Class to explain (``1`` = genuine, ``0`` = impostor).
            ``None`` auto-selects the predicted class.
        face_layer: Name of the face CNN feature layer to target.
        fp_layer: Name of the fingerprint CNN feature layer to target.

    Returns:
        ``(face_heatmap, fingerprint_heatmap)`` as ``(H, W)`` numpy arrays
        in ``[0, 1]``.

    Example:
        >>> face_hm, fp_hm = compute_gradcam_bimodal(
        ...     model, face_img, fp_img, target_class=1
        ... )
    """
    gradcam = GradCAMBimodal(model, face_layer, fp_layer)
    try:
        face_hm, fp_hm = gradcam(face_img, fp_img, target_class)
    finally:
        gradcam.release()
    return face_hm, fp_hm


# ---------------------------------------------------------------------------
# Grad-CAM overlay visualisation
# ---------------------------------------------------------------------------


def visualize_gradcam_bimodal(
    face_img: torch.Tensor,
    fp_img: torch.Tensor,
    face_heatmap: np.ndarray,
    fp_heatmap: np.ndarray,
    save_path: Union[str, Path],
    face_alpha: float = 0.5,
    fp_alpha: float = 0.5,
    title: Optional[str] = None,
) -> None:
    """Overlay Grad-CAM heatmaps on original images.

    Creates a 4-panel figure:
        1. Original face image
        2. Face image with Grad-CAM overlay (eyes, nose highlighted)
        3. Original fingerprint image
        4. Fingerprint image with Grad-CAM overlay (ridges, minutiae)

    Args:
        face_img: Face image tensor ``(B, C, H, W)`` or ``(C, H, W)``.
        fp_img: Fingerprint image tensor ``(B, C, H, W)`` or ``(C, H, W)``.
        face_heatmap: Grad-CAM heatmap for face ``(H, W)`` in ``[0, 1]``.
        fp_heatmap: Grad-CAM heatmap for fingerprint ``(H, W)`` in ``[0, 1]``.
        save_path: Output PDF path.
        face_alpha: Opacity of the face heatmap overlay.
        fp_alpha: Opacity of the fingerprint heatmap overlay.
        title: Optional figure title.

    Returns:
        ``None`` (figure saved to disk).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare display images
    face_np = _tensor_to_display(_ensure_single(face_img))  # (H, W, 3)
    fp_np = _tensor_to_display(_ensure_single(fp_img))  # (H, W, 3)

    # Resize heatmaps to match image dimensions
    face_hm_resized = _resize_heatmap(face_heatmap, face_np.shape[:2])
    fp_hm_resized = _resize_heatmap(fp_heatmap, fp_np.shape[:2])

    # Colourise heatmaps
    face_hm_colour = _apply_colourmap(face_hm_resized, _CMAP_FACE)
    fp_hm_colour = _apply_colourmap(fp_hm_resized, _CMAP_FP)

    # Blend overlays
    face_overlay = _blend_images(face_np, face_hm_colour, face_alpha)
    fp_overlay = _blend_images(fp_np, fp_hm_colour, fp_alpha)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(_FIG_WIDTH, _FIG_HEIGHT))

    axes[0, 0].imshow(face_np)
    axes[0, 0].set_title("Original Face", fontsize=11)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(face_overlay)
    axes[0, 1].set_title("Face Grad-CAM", fontsize=11)
    axes[0, 1].axis("off")

    axes[1, 0].imshow(fp_np, cmap="gray")
    axes[1, 0].set_title("Original Fingerprint", fontsize=11)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(fp_overlay)
    axes[1, 1].set_title("Fingerprint Grad-CAM", fontsize=11)
    axes[1, 1].axis("off")

    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight", format="pdf")
    plt.close(fig)

    logging.info("Grad-CAM visualisation saved to %s", save_path)


# ---------------------------------------------------------------------------
# Attention heatmap plotting
# ---------------------------------------------------------------------------


def plot_attention_heatmap(
    attention_matrix: np.ndarray,
    save_path: Union[str, Path],
    xlabel: str = "Fingerprint patches",
    ylabel: str = "Face patches",
    title: str = "Cross-Modal Attention Matrix",
    figsize: Tuple[float, float] = (10, 8),
) -> None:
    """Plot cross-modal attention matrix as a heatmap.

    Rows correspond to face patches (queries) and columns to fingerprint
    patches (keys). Higher values indicate stronger attention from a face
    region to a fingerprint region.

    Args:
        attention_matrix: 2-D numpy array ``(num_face_patches,
            num_fp_patches)``.
        save_path: Output PDF path.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        title: Figure title.
        figsize: Figure size in inches.

    Returns:
        ``None`` (figure saved to disk).

    Example:
        >>> attn = np.random.rand(196, 196)
        >>> plot_attention_heatmap(attn, "attention_matrix.pdf")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(attention_matrix, cmap=_CMAP_ATTN, aspect="auto",
                   interpolation="nearest")

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", fontsize=10)

    # Add statistics as text
    mean_attn = np.mean(attention_matrix)
    max_attn = np.max(attention_matrix)
    min_attn = np.min(attention_matrix)
    stats_text = (
        f"Mean: {mean_attn:.4f}  |  Max: {max_attn:.4f}  |  Min: {min_attn:.4f}"
    )
    ax.text(0.5, -0.12, stats_text, transform=ax.transAxes,
            ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight", format="pdf")
    plt.close(fig)

    logging.info("Attention heatmap saved to %s", save_path)


# ---------------------------------------------------------------------------
# Explainability report generation
# ---------------------------------------------------------------------------


def generate_explainability_report(
    model: nn.Module,
    test_pairs: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    device: Optional[torch.device] = None,
    max_identities: int = 10,
    face_layer: str = "face_cnn.layer4",
    fp_layer: str = "fp_cnn.layer4",
) -> Path:
    """Generate a multi-page PDF explainability report for test identities.

    For each of ``max_identities`` test subjects, the report contains:

    * A genuine pair (same identity) with Grad-CAM overlays showing which
      facial regions and fingerprint minutiae support the *genuine* decision.
    * An impostor pair (different identities) with Grad-CAM overlays showing
      regions supporting the *impostor* decision.
    * Cross-modal attention heatmaps for both pairs.

    Args:
        model: Trained UFM-Transformer model.
        test_pairs: List of dictionaries, each with keys:

            - ``"face1"``: Face image tensor ``(C, H, W)``
            - ``"face2"``: Second face image tensor
            - ``"fp1"``: Fingerprint image tensor
            - ``"fp2"``: Second fingerprint image tensor
            - ``"label"``: ``1`` for genuine, ``0`` for impostor
            - ``"identity"``: String identity label

        output_dir: Directory to save report and intermediate figures.
        device: Compute device. ``None`` uses CUDA if available.
        max_identities: Number of identities to include in the report.
        face_layer: Face CNN layer name for Grad-CAM.
        fp_layer: Fingerprint CNN layer name for Grad-CAM.

    Returns:
        Path to the generated multi-page PDF report.

    Example:
        >>> report_path = generate_explainability_report(
        ...     model, test_pairs, output_dir="./reports"
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "explainability_report.pdf"

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()
    gradcam = GradCAMBimodal(model, face_layer, fp_layer)

    # Group pairs by identity
    identity_groups: Dict[str, List[Dict[str, Any]]] = {}
    for pair in test_pairs:
        identity = pair.get("identity", "unknown")
        identity_groups.setdefault(identity, []).append(pair)

    # Select top identities
    selected_identities = list(identity_groups.keys())[:max_identities]

    with PdfPages(report_path) as pdf:
        # Title page
        _add_report_title_page(pdf, len(selected_identities))

        for identity in selected_identities:
            pairs = identity_groups[identity]
            genuine_pairs = [p for p in pairs if p.get("label", 0) == 1]
            impostor_pairs = [p for p in pairs if p.get("label", 0) == 0]

            # Pick one genuine and one impostor pair if available
            pair_genuine = genuine_pairs[0] if genuine_pairs else None
            pair_impostor = impostor_pairs[0] if impostor_pairs else None

            # --- Grad-CAM page ---
            fig = _create_gradcam_page(
                model, gradcam, identity, pair_genuine, pair_impostor, device
            )
            if fig is not None:
                pdf.savefig(fig, dpi=_DPI)
                plt.close(fig)

            # --- Attention heatmap page ---
            fig = _create_attention_page(
                model, identity, pair_genuine, pair_impostor, device
            )
            if fig is not None:
                pdf.savefig(fig, dpi=_DPI)
                plt.close(fig)

    gradcam.release()
    logging.info("Explainability report saved to %s", report_path)
    return report_path


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch image tensor to a displayable numpy RGB array.

    Handles normalisation by shifting to [0, 1] range if values are
    outside ``[0, 1]`` (assumes ImageNet normalisation: mean≈0.45, std≈0.22).

    Args:
        tensor: Image tensor ``(C, H, W)`` or ``(1, C, H, W)``.

    Returns:
        Numpy uint8 RGB array ``(H, W, 3)``.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)  # (H, W, C)

    # If normalised (values outside [0, 1]), denormalise
    if img.min() < 0 or img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img


def _ensure_single(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is single-sample ``(C, H, W)`` not batched."""
    if tensor.dim() == 4:
        return tensor[0]
    return tensor


def _resize_heatmap(heatmap: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize heatmap to target ``(H, W)`` using OpenCV."""
    resized = cv2.resize(heatmap, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_LINEAR)
    return resized


def _apply_colourmap(heatmap: np.ndarray, cmap_name: str) -> np.ndarray:
    """Apply matplotlib colormap to a grayscale heatmap, return RGB uint8."""
    cmap = matplotlib.colormaps[cmap_name]
    heatmap_norm = np.clip(heatmap, 0, 1)
    heatmap_colour = cmap(heatmap_norm)[:, :, :3]  # Drop alpha
    heatmap_colour = (heatmap_colour * 255).astype(np.uint8)
    return heatmap_colour


def _blend_images(
    base: np.ndarray,
    overlay: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Alpha-blend overlay onto base image. Both are uint8 RGB."""
    return cv2.addWeighted(base, 1.0 - alpha, overlay, alpha, 0)


def _add_report_title_page(pdf: PdfPages, n_identities: int) -> None:
    """Add a title page to the PDF report."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")

    title_text = (
        "UFM-Transformer\n"
        "Explainability Report\n\n"
        "Cross-Modal Biometric Verification\n"
        f"Face + Fingerprint Attention & Grad-CAM\n\n"
        f"Number of Identities: {n_identities}\n\n"
        "For each identity:\n"
        "  - Genuine pair Grad-CAM\n"
        "  - Impostor pair Grad-CAM\n"
        "  - Cross-modal attention heatmaps\n\n"
        "\u00a9 UFM-Transformer Project"
    )
    ax.text(0.5, 0.5, title_text, transform=ax.transAxes,
            fontsize=16, ha="center", va="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.3))
    pdf.savefig(fig, dpi=_DPI)
    plt.close(fig)


def _create_gradcam_page(
    model: nn.Module,
    gradcam: GradCAMBimodal,
    identity: str,
    pair_genuine: Optional[Dict[str, Any]],
    pair_impostor: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[plt.Figure]:
    """Create a 4-panel Grad-CAM figure for genuine and impostor pairs."""
    has_genuine = pair_genuine is not None
    has_impostor = pair_impostor is not None

    if not has_genuine and not has_impostor:
        return None

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Identity: {identity} — Grad-CAM Explanations",
                 fontsize=14, fontweight="bold")

    # --- Genuine pair ---
    if has_genuine:
        pg = pair_genuine
        f1 = pg["face1"].unsqueeze(0).to(device)
        f2 = pg["face2"].unsqueeze(0).to(device)
        fp1 = pg["fp1"].unsqueeze(0).to(device)
        fp2 = pg["fp2"].unsqueeze(0).to(device)

        face_hm_g, fp_hm_g = gradcam(f1, fp1, target_class=1)
        face_hm_g2, fp_hm_g2 = gradcam(f2, fp2, target_class=1)

        _plot_single_gradcam(axes[0, 0], _tensor_to_display(f1[0]),
                             face_hm_g, "Genuine: Face 1")
        _plot_single_gradcam(axes[0, 1], _tensor_to_display(fp1[0]),
                             fp_hm_g, "Genuine: FP 1", is_gray=True)
        _plot_single_gradcam(axes[0, 2], _tensor_to_display(f2[0]),
                             face_hm_g2, "Genuine: Face 2")
        _plot_single_gradcam(axes[0, 3], _tensor_to_display(fp2[0]),
                             fp_hm_g2, "Genuine: FP 2", is_gray=True)
    else:
        for c in range(4):
            axes[0, c].axis("off")
            axes[0, c].set_title("No genuine pair", fontsize=10)

    # --- Impostor pair ---
    if has_impostor:
        pi = pair_impostor
        f1 = pi["face1"].unsqueeze(0).to(device)
        fp1 = pi["fp1"].unsqueeze(0).to(device)
        f2 = pi["face2"].unsqueeze(0).to(device)
        fp2 = pi["fp2"].unsqueeze(0).to(device)

        face_hm_i, fp_hm_i = gradcam(f1, fp1, target_class=0)
        face_hm_i2, fp_hm_i2 = gradcam(f2, fp2, target_class=0)

        _plot_single_gradcam(axes[1, 0], _tensor_to_display(f1[0]),
                             face_hm_i, "Impostor: Face 1")
        _plot_single_gradcam(axes[1, 1], _tensor_to_display(fp1[0]),
                             fp_hm_i, "Impostor: FP 1", is_gray=True)
        _plot_single_gradcam(axes[1, 2], _tensor_to_display(f2[0]),
                             face_hm_i2, "Impostor: Face 2")
        _plot_single_gradcam(axes[1, 3], _tensor_to_display(fp2[0]),
                             fp_hm_i2, "Impostor: FP 2", is_gray=True)
    else:
        for c in range(4):
            axes[1, c].axis("off")
            axes[1, c].set_title("No impostor pair", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def _plot_single_gradcam(
    ax: plt.Axes,
    img_np: np.ndarray,
    heatmap: np.ndarray,
    title: str,
    is_gray: bool = False,
    alpha: float = 0.5,
) -> None:
    """Plot a single image with Grad-CAM overlay on the given axes."""
    hm_resized = _resize_heatmap(heatmap, img_np.shape[:2])
    hm_colour = _apply_colourmap(hm_resized, _CMAP_FACE if not is_gray else _CMAP_FP)
    overlay = _blend_images(img_np, hm_colour, alpha)

    ax.imshow(overlay)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _create_attention_page(
    model: nn.Module,
    identity: str,
    pair_genuine: Optional[Dict[str, Any]],
    pair_impostor: Optional[Dict[str, Any]],
    device: torch.device,
) -> Optional[plt.Figure]:
    """Create attention heatmap figures for genuine and impostor pairs."""
    has_genuine = pair_genuine is not None
    has_impostor = pair_impostor is not None

    if not has_genuine and not has_impostor:
        return None

    ncols = (1 if has_genuine else 0) + (1 if has_impostor else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)
    fig.suptitle(f"Identity: {identity} — Cross-Modal Attention",
                 fontsize=14, fontweight="bold")

    col_idx = 0
    if has_genuine:
        pg = pair_genuine
        f1 = pg["face1"].unsqueeze(0).to(device)
        fp1 = pg["fp1"].unsqueeze(0).to(device)
        attn = extract_attention_maps(model, f1, fp1)

        if attn:
            # Use first cross-attention layer, average over heads
            first_key = list(attn.keys())[0]
            attn_mat = attn[first_key][0].mean(dim=0).numpy()  # (seq_q, seq_k)
            im = axes[0, col_idx].imshow(attn_mat, cmap=_CMAP_ATTN, aspect="auto")
            axes[0, col_idx].set_title("Genuine Pair", fontsize=11)
            axes[0, col_idx].set_xlabel("Fingerprint patches")
            axes[0, col_idx].set_ylabel("Face patches")
            plt.colorbar(im, ax=axes[0, col_idx], fraction=0.046)
        else:
            axes[0, col_idx].text(0.5, 0.5, "No attention maps",
                                  ha="center", va="center")
        col_idx += 1

    if has_impostor:
        pi = pair_impostor
        f1 = pi["face1"].unsqueeze(0).to(device)
        fp1 = pi["fp1"].unsqueeze(0).to(device)
        attn = extract_attention_maps(model, f1, fp1)

        if attn:
            first_key = list(attn.keys())[0]
            attn_mat = attn[first_key][0].mean(dim=0).numpy()
            im = axes[0, col_idx].imshow(attn_mat, cmap=_CMAP_ATTN, aspect="auto")
            axes[0, col_idx].set_title("Impostor Pair", fontsize=11)
            axes[0, col_idx].set_xlabel("Fingerprint patches")
            axes[0, col_idx].set_ylabel("Face patches")
            plt.colorbar(im, ax=axes[0, col_idx], fraction=0.046)
        else:
            axes[0, col_idx].text(0.5, 0.5, "No attention maps",
                                  ha="center", va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Running visualization module self-test...")

    # 1. Test attention heatmap plotting
    dummy_attn = np.random.rand(196, 196).astype(np.float32)
    dummy_attn = (dummy_attn + dummy_attn.T) / 2  # Symmetrise
    plot_attention_heatmap(dummy_attn, "/tmp/test_attention_heatmap.pdf",
                           title="Test Attention Matrix")
    print("[OK] plot_attention_heatmap")

    # 2. Test Grad-CAM overlay visualisation
    face_dummy = torch.rand(3, 224, 224)
    fp_dummy = torch.rand(3, 224, 224)
    face_hm = np.random.rand(224, 224).astype(np.float32)
    fp_hm = np.random.rand(224, 224).astype(np.float32)

    visualize_gradcam_bimodal(
        face_dummy, fp_dummy, face_hm, fp_hm,
        "/tmp/test_gradcam_overlay.pdf",
        title="Test Grad-CAM Overlay",
    )
    print("[OK] visualize_gradcam_bimodal")

    # 3. Test cross-attention visualisation
    dummy_attn_dict = {
        "cross_attn_1": torch.softmax(torch.randn(2, 8, 196, 196), dim=-1),
        "cross_attn_2": torch.softmax(torch.randn(2, 4, 196, 196), dim=-1),
    }
    face_batch = torch.rand(2, 3, 224, 224)
    fp_batch = torch.rand(2, 3, 224, 224)
    visualize_cross_attention(
        dummy_attn_dict, face_batch, fp_batch,
        "/tmp/test_cross_attention.pdf",
    )
    print("[OK] visualize_cross_attention")

    # 4. Clean up temp files
    for f in ["/tmp/test_attention_heatmap.pdf",
              "/tmp/test_gradcam_overlay.pdf",
              "/tmp/test_cross_attention.pdf"]:
        if os.path.exists(f):
            os.remove(f)

    print("All visualization tests passed.")
