"""
evaluate_separate.py — Automated Unimodal Evaluation for Separate-Dataset Mode.

When CASIA-WebFace (faces) and SOCOFing (fingerprints) are trained independently
(USE_SEPARATE=True), there is no shared test set for multimodal evaluation.
This script evaluates each encoder independently on a held-out validation split
of its own dataset and produces the same figures/ + tables/ structure expected by
the LaTeX copy step in the notebook.

Protocol:
  - Face:        extract embeddings with UFMTransformer.face_encoder + face_projector
                 on the last VAL_SPLIT fraction of CASIA-WebFace subjects.
  - Fingerprint: extract embeddings with UFMTransformer.fp_encoder + fp_projector
                 on the last VAL_SPLIT fraction of SOCOFing subjects.
  - For each modality: generate genuine/impostor pairs, compute EER/TAR@FAR/AUC,
    plot ROC + DET + score distribution, export a LaTeX metrics table.

Usage::

    python evaluate_separate.py \\
        --face_path /path/to/casia-webface-extracted \\
        --fingerprint_path /path/to/SOCOFing/Real \\
        --model_path /kaggle/working/output/best_model_phase2.pt \\
        --output_dir /kaggle/working/results \\
        --device auto

Author: Biometrics Pipeline Team
Python Version: >=3.10
PyTorch Version: >=2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

matplotlib.use("Agg")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.figsize"] = (8, 6)

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data_loader import UnimodalFaceDataset, UnimodalFingerprintDataset
from models import UFMTransformerModel

# Re-use metric and plotting functions from evaluate.py
from evaluate import (
    compute_eer,
    compute_tar_at_far,
    compute_auc,
    compute_det_curve,
    compute_all_metrics,
    plot_roc_curve,
    plot_det_curve,
    plot_score_distributions,
    export_metrics_latex,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VAL_SPLIT = 0.20          # Fraction of subjects held out for evaluation
MAX_GENUINE_PER_SUBJ = 15  # Cap genuine pairs per subject
MAX_IMPOSTORS = 5000       # Total impostor pairs
C_GEN = "#2E86AB"
C_IMP = "#A23B72"

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device, embed_dim: int) -> nn.Module:
    """Load a UFMTransformer checkpoint and return the unwrapped model."""
    checkpoint_path = Path(model_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    model = UFMTransformerModel(embed_dim=embed_dim)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)

    # Support different checkpoint formats
    if isinstance(checkpoint, dict):
        state = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state = checkpoint

    # Strip DDP prefix if present
    clean_state = {}
    for k, v in state.items():
        clean_state[k.replace("module.", "", 1)] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if missing:
        warnings.warn(f"Missing keys when loading checkpoint: {missing[:5]}...")
    if unexpected:
        warnings.warn(f"Unexpected keys in checkpoint: {unexpected[:5]}...")

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Held-out validation split
# ---------------------------------------------------------------------------

def _make_val_dataset(
    dataset: UnimodalFaceDataset | UnimodalFingerprintDataset,
    val_split: float = VAL_SPLIT,
) -> Subset:
    """Return a Subset containing the last `val_split` fraction of subjects.

    Subjects are sorted deterministically (the datasets already sort them),
    so the same subjects are always in validation across runs.
    """
    # Collect unique subject IDs in the order they appear in `dataset.samples`
    seen: dict[int, None] = {}
    for _, sid in dataset.samples:
        seen[sid] = None
    all_subjects = list(seen.keys())  # insertion-ordered = dataset's sort order

    n_val_subjects = max(1, int(len(all_subjects) * val_split))
    val_subjects = set(all_subjects[-n_val_subjects:])  # last N subjects

    val_indices = [
        i for i, (_, sid) in enumerate(dataset.samples)
        if sid in val_subjects
    ]
    return Subset(dataset, val_indices)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,          # "face" or "fingerprint"
    desc: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract L2-normalised embeddings and subject IDs from a DataLoader.

    Returns:
        (embeddings, subject_ids) — numpy arrays of shape (N, D) and (N,).
    """
    embeddings_list: List[torch.Tensor] = []
    sids_list: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc or f"Extracting {modality}", leave=False):
            images, subject_ids, _ = batch  # (img, sid, quality)
            images = images.to(device, non_blocking=True)

            if modality == "face":
                feat_map = model.face_encoder(images)          # (B, C, H, W)
                proj = model.face_projector                    # Projector module
            else:
                feat_map = model.fp_encoder(images)            # (B, C, H, W)
                proj = model.fp_projector

            # Global-average pool the feature map, then project to embed space
            # Shape: (B, C, H, W) -> (B, C) via adaptive avg pool
            if feat_map.dim() == 4:
                feat_vec = F.adaptive_avg_pool2d(feat_map, (1, 1)).flatten(1)  # (B, C)
            else:
                feat_vec = feat_map  # already (B, C)

            z = proj(feat_vec)                                  # (B, embed_dim)
            z = F.normalize(z, p=2, dim=1)                     # L2-normalise

            embeddings_list.append(z.cpu())
            sids_list.extend(subject_ids.tolist() if hasattr(subject_ids, "tolist")
                             else list(subject_ids))

    embeddings = torch.cat(embeddings_list, dim=0).numpy()  # (N, D)
    subject_ids_np = np.array(sids_list, dtype=np.int64)    # (N,)
    return embeddings, subject_ids_np


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _generate_pairs(
    embeddings: np.ndarray,
    subject_ids: np.ndarray,
    max_genuine_per_subj: int = MAX_GENUINE_PER_SUBJ,
    max_impostors: int = MAX_IMPOSTORS,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate genuine and impostor cosine-similarity scores.

    Returns:
        (scores_genuine, scores_impostor) — 1D float arrays.
    """
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)

    subject_indices: Dict[int, List[int]] = {s: [] for s in unique_subjects}
    for idx, sid in enumerate(subject_ids):
        subject_indices[int(sid)].append(idx)

    genuine_scores: List[float] = []
    for sid in unique_subjects:
        idxs = subject_indices[int(sid)]
        if len(idxs) < 2:
            continue
        # Generate all pairs up to the cap
        pairs_done = 0
        used = set()
        idxs_arr = np.array(idxs)
        for _ in range(max_genuine_per_subj * 10):
            if pairs_done >= max_genuine_per_subj:
                break
            i, j = rng.choice(len(idxs_arr), size=2, replace=False)
            key = (min(i, j), max(i, j))
            if key in used:
                continue
            used.add(key)
            sim = float(np.dot(embeddings[idxs_arr[i]], embeddings[idxs_arr[j]]))
            genuine_scores.append(sim)
            pairs_done += 1

    impostor_scores: List[float] = []
    subj_list = list(unique_subjects)
    for _ in range(max_impostors * 5):
        if len(impostor_scores) >= max_impostors:
            break
        s1, s2 = rng.choice(len(subj_list), size=2, replace=False)
        sid_a, sid_b = subj_list[s1], subj_list[s2]
        if sid_a == sid_b:
            continue
        ia = rng.choice(subject_indices[int(sid_a)])
        ib = rng.choice(subject_indices[int(sid_b)])
        sim = float(np.dot(embeddings[ia], embeddings[ib]))
        impostor_scores.append(sim)

    return np.array(genuine_scores, dtype=np.float64), np.array(impostor_scores, dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-modality evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_modality(
    model: nn.Module,
    dataset: UnimodalFaceDataset | UnimodalFingerprintDataset,
    modality: str,
    device: torch.device,
    figures_dir: str,
    tables_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    embed_dim: int = 512,
) -> Dict[str, float]:
    """Run the full unimodal evaluation pipeline for one modality.

    Args:
        model:         Trained UFMTransformerModel.
        dataset:       Full unimodal dataset (train + val subjects).
        modality:      "face" or "fingerprint".
        device:        Compute device.
        figures_dir:   Output directory for PDF figures.
        tables_dir:    Output directory for LaTeX tables.
        batch_size:    Batch size for embedding extraction.
        num_workers:   DataLoader workers.
        embed_dim:     Expected embedding dimension (for logging only).

    Returns:
        Metrics dictionary with EER, AUC, TAR@FAR, etc.
    """
    label = modality.capitalize()
    print(f"\n{'='*60}")
    print(f" {label} Unimodal Evaluation")
    print(f"{'='*60}")

    # --- Held-out validation subset ---
    val_subset = _make_val_dataset(dataset, val_split=VAL_SPLIT)
    n_val = len(val_subset)
    n_total = len(dataset)
    n_subjects_val = len({dataset.samples[i][1] for i in val_subset.indices})

    print(f"  Dataset:      {n_total} images  |  held-out: {n_val} images "
          f"({n_subjects_val} subjects, {VAL_SPLIT*100:.0f}% split)")

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # --- Extract embeddings ---
    embeddings, subject_ids = _extract_embeddings(
        model, val_loader, device, modality, desc=f"Extracting {label} embeddings"
    )
    print(f"  Extracted:    {embeddings.shape[0]} embeddings  (dim={embeddings.shape[1]})")

    # --- Generate pairs ---
    gen_scores, imp_scores = _generate_pairs(embeddings, subject_ids)
    if len(gen_scores) == 0 or len(imp_scores) == 0:
        warnings.warn(f"Not enough pairs for {label} evaluation (genuine={len(gen_scores)}, "
                      f"impostor={len(imp_scores)}). Returning default metrics.")
        return {"eer": 50.0, "auc": 0.0, "tar_at_0_001": 0.0, "tar_at_0_01": 0.0}

    print(f"  Pairs:        {len(gen_scores)} genuine  |  {len(imp_scores)} impostor")

    # --- Compute metrics ---
    metrics = compute_all_metrics(gen_scores, imp_scores)
    print(f"\n  EER:          {metrics.get('eer', 0):.3f}%")
    print(f"  AUC:          {metrics.get('auc', 0):.3f}%")
    print(f"  TAR@FAR=1%:   {metrics.get('tar_at_0_01', 0):.3f}%")
    print(f"  TAR@FAR=0.1%: {metrics.get('tar_at_0_001', 0):.3f}%")

    # --- Figures ---
    tag = modality  # "face" or "fingerprint"
    plot_roc_curve(gen_scores, imp_scores,
                   os.path.join(figures_dir, f"roc_curve_{tag}.pdf"))
    print(f"  Saved:        figures/roc_curve_{tag}.pdf")

    fmr, fnmr, _ = compute_det_curve(gen_scores, imp_scores)
    plot_det_curve(fmr, fnmr,
                   os.path.join(figures_dir, f"det_curve_{tag}.pdf"))
    print(f"  Saved:        figures/det_curve_{tag}.pdf")

    plot_score_distributions(gen_scores, imp_scores,
                              os.path.join(figures_dir, f"score_distributions_{tag}.pdf"))
    print(f"  Saved:        figures/score_distributions_{tag}.pdf")

    # --- LaTeX table ---
    export_metrics_latex(
        metrics,
        os.path.join(tables_dir, f"metrics_{tag}.tex"),
        model_name=f"UFM-Transformer ({label} Encoder)",
    )
    print(f"  Saved:        tables/metrics_{tag}.tex")

    return metrics


# ---------------------------------------------------------------------------
# Checkpoint selection helper
# ---------------------------------------------------------------------------

def _best_checkpoint(output_dir: Path) -> Optional[Path]:
    """Return the best available checkpoint path, preferring Phase 2."""
    candidates = [
        output_dir / "best_model_phase2.pt",
        output_dir / "best_model_phase1.pt",
        output_dir / "shutdown_checkpoint_phase2.pt",
        output_dir / "shutdown_checkpoint_phase1.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fall back to any .pt file in the directory
    pts = sorted(output_dir.glob("*.pt"))
    return pts[-1] if pts else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unimodal evaluation for UFM-Transformer trained in separate-dataset mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--face_path", type=str, required=True,
                        help="Root directory of CASIA-WebFace extracted.")
    parser.add_argument("--fingerprint_path", type=str, required=True,
                        help="Root directory of SOCOFing/Real.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to checkpoint (.pt). Auto-detected if omitted.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory where figures/ and tables/ will be written.")
    parser.add_argument("--embed_dim", type=int, default=512,
                        help="Embedding dimension used when training the model.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image size for preprocessing.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for embedding extraction.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker count.")
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT,
                        help="Fraction of subjects to hold out for evaluation.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device: auto | cuda | cpu.")
    args = parser.parse_args()

    device = _get_device(args.device)
    output_dir = Path(args.output_dir)
    figures_dir = str(output_dir / "figures")
    tables_dir = str(output_dir / "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 60)
    print(" UFM-Transformer: Unimodal Evaluation (Separate Mode)")
    print("=" * 60)
    print(f"  Device:       {device}")
    print(f"  Output:       {output_dir}")

    # --- Resolve checkpoint ---
    if args.model_path:
        ckpt_path = Path(args.model_path)
    else:
        ckpt_path = _best_checkpoint(output_dir)
        if ckpt_path is None:
            # try parent directory (e.g. /kaggle/working/output)
            ckpt_path = _best_checkpoint(output_dir.parent)

    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found. Provide --model_path or ensure a .pt file "
            f"exists in {output_dir} or {output_dir.parent}."
        )
    print(f"  Checkpoint:   {ckpt_path}")

    # --- Load model ---
    model = _load_model(str(ckpt_path), device, embed_dim=args.embed_dim)

    # --- Build full datasets (is_training=False, no quality degradation) ---
    print("\n[1] Loading datasets...")
    face_dataset = UnimodalFaceDataset(
        root_dir=args.face_path,
        image_size=args.image_size,
        is_training=False,
        apply_quality_degradation=False,
    )
    fp_dataset = UnimodalFingerprintDataset(
        root_dir=args.fingerprint_path,
        image_size=args.image_size,
        is_training=False,
        apply_quality_degradation=False,
    )
    print(f"  Face dataset:        {len(face_dataset)} images | "
          f"{face_dataset.get_num_subjects()} subjects")
    print(f"  Fingerprint dataset: {len(fp_dataset)} images | "
          f"{fp_dataset.get_num_subjects()} subjects")

    # --- Run per-modality evaluation ---
    print("\n[2] Evaluating face encoder...")
    face_metrics = evaluate_modality(
        model=model,
        dataset=face_dataset,
        modality="face",
        device=device,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embed_dim=args.embed_dim,
    )

    print("\n[3] Evaluating fingerprint encoder...")
    fp_metrics = evaluate_modality(
        model=model,
        dataset=fp_dataset,
        modality="fingerprint",
        device=device,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        embed_dim=args.embed_dim,
    )

    # --- Combined summary table ---
    print("\n[4] Exporting combined comparison table...")
    _export_combined_table(
        face_metrics=face_metrics,
        fp_metrics=fp_metrics,
        save_path=os.path.join(tables_dir, "metrics_unimodal_comparison.tex"),
    )
    print("  Saved: tables/metrics_unimodal_comparison.tex")

    # --- Save JSON summary ---
    summary = {
        "face": face_metrics,
        "fingerprint": fp_metrics,
        "_note": "Unimodal evaluation on held-out validation subjects (separate-dataset mode)",
    }
    json_path = output_dir / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {json_path}")

    # --- Final printout ---
    print("\n" + "=" * 60)
    print(" Summary")
    print("=" * 60)
    header = f"{'Modality':<15} {'EER':>8} {'AUC':>8} {'TAR@1%':>10} {'TAR@0.1%':>12}"
    print(header)
    print("-" * 60)
    for name, m in [("Face", face_metrics), ("Fingerprint", fp_metrics)]:
        row = (f"{name:<15}"
               f" {m.get('eer', 0):>7.2f}%"
               f" {m.get('auc', 0):>7.2f}%"
               f" {m.get('tar_at_0_01', 0):>9.2f}%"
               f" {m.get('tar_at_0_001', 0):>11.2f}%")
        print(row)
    print("=" * 60)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"  figures/ : {len(list(Path(figures_dir).glob('*.pdf')))} PDFs")
    print(f"  tables/  : {len(list(Path(tables_dir).glob('*.tex')))} .tex files")
    print("\nEvaluation complete!")


# ---------------------------------------------------------------------------
# Combined table helper
# ---------------------------------------------------------------------------

def _export_combined_table(
    face_metrics: Dict[str, float],
    fp_metrics: Dict[str, float],
    save_path: str,
) -> None:
    """Export a side-by-side LaTeX table comparing face and fingerprint metrics."""
    display = [
        ("eer",          "EER (\\%)",           "{:.2f}"),
        ("tar_at_0_01",  "TAR @ FAR=1\\%",      "{:.2f}"),
        ("tar_at_0_001", "TAR @ FAR=0.1\\%",    "{:.2f}"),
        ("auc",          "AUC-ROC (\\%)",        "{:.2f}"),
    ]

    lines = [
        "% Unimodal evaluation comparison (separate-dataset mode)",
        "% Generated by evaluate_separate.py",
        "",
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Unimodal Verification Performance (Separate Dataset Mode)}",
        r"    \label{tab:unimodal_comparison}",
        r"    \begin{tabular}{lcc}",
        r"        \toprule",
        r"        \textbf{Metric} & \textbf{Face Encoder} & \textbf{Fingerprint Encoder} \\",
        r"        \midrule",
    ]
    for key, display_name, fmt in display:
        fv = fmt.format(face_metrics.get(key, 0.0))
        pv = fmt.format(fp_metrics.get(key, 0.0))
        lines.append(
            f"        {display_name} & "
            f"\\textcolor{{red}}{{\\textbf{{{fv}}}}} & "
            f"\\textcolor{{red}}{{\\textbf{{{pv}}}}} \\\\"
        )
    lines += [
        r"        \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
        "",
    ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
