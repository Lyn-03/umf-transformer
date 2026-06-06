# UFM-Transformer: Uncertainty-aware Fusion with Missing-modality Transformer

> **Unified Face and Fingerprint Multimodal Biometric Verification**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table des matieres / Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [Training](#1-training-mode)
  - [Evaluation](#2-evaluation-mode)
  - [Visualization](#3-visualization-mode)
- [Project Structure](#project-structure)
- [Hyperparameters](#hyperparameters)
- [Expected Outputs](#expected-outputs)
- [Citation](#citation)
- [License](#license)

---

## Overview

The **UFM-Transformer** (*Uncertainty-aware Fusion with Missing-modality Transformer*) is a deep learning architecture for multimodal biometric verification that jointly processes **face** and **fingerprint** modalities. The model addresses three critical challenges in biometric systems:

1. **Cross-modal fusion**: A quality-aware cross-modal transformer dynamically fuses face and fingerprint representations, attending to the most discriminative regions of each modality.
2. **Missing modality handling**: Learnable tokens gracefully handle scenarios where one modality is unavailable (e.g., occluded face or poor-quality fingerprint).
3. **Uncertainty quantification**: Monte-Carlo Dropout decomposes predictive uncertainty into epistemic (model) and aleatoric (data) components, enabling trust-aware decision-making.

The architecture uses **EfficientNet-B2** as the face encoder and a custom **ResNet-style CNN** for fingerprint ridge-pattern extraction. Features are projected into a common embedding space, fused via multi-head cross-attention, and scored using an ArcFace-style similarity head.

### Key Features

| Feature | Description |
|---------|-------------|
| **2-Phase Training** | Unimodal pre-training (Phase 1) + Joint fine-tuning with modality dropout (Phase 2) |
| **Composite Loss** | Triplet loss (hard negative mining) + ArcFace margin softmax + Uncertainty regularization |
| **5 Baseline Models** | ConcatenationFusion, ScoreSumFusion, DenseNetBiModal, TransformerFusionNoUncertainty, MDRLFusion |
| **7 Evaluation Plots** | ROC curves, DET curves, EER histograms, confusion matrices, calibration plots, similarity distributions, FNMR/FMR curves |
| **LaTeX Export** | Publication-ready tables exported directly to `.tex` files |
| **Explainability** | Attention heatmaps, cross-modal attention matrices, Grad-CAM overlays |

### Dimension Flow (Batch size = B)

```
Face image:      (B, 3, 224, 224)
FP image:        (B, 1, 224, 224)
Face feat map:   (B, 1408, 7, 7)   <- EfficientNet-B2 conv features
FP feat map:     (B, 512, 7, 7)     <- Custom CNN conv features
Face global:     (B, 1408)
FP global:       (B, 512)
Projected:       (B, 256)           <- Common embedding space (L2-normalized)
Fused:           (B, 256)           <- After cross-modal transformer
Similarity:      (B,)               <- Verification score [-1, 1]
Uncertainty:     (B,)               <- Total predictive uncertainty
```

---

## Architecture

The UFM-Transformer model (`models.py`) comprises **10 PyTorch classes**:

| # | Class | Description |
|---|-------|-------------|
| 1 | `FaceEncoder` | EfficientNet-B2 feature extractor (ImageNet-pretrained) |
| 2 | `ResidualBlock` | 3x3 conv residual block for fingerprint CNN |
| 3 | `FingerprintEncoder` | Custom ResNet-style CNN for fingerprint ridge patterns |
| 4 | `QualityEstimator` | Heuristic quality scoring (Laplacian variance + contrast) |
| 5 | `Projector` | L2-normalized projection to common embedding space |
| 6 | `LearnableToken` | Learnable placeholder token for missing modalities |
| 7 | `CrossAttentionLayer` | Single cross-modal attention layer with quality gating |
| 8 | `CrossModalTransformer` | Stack of cross-attention layers for fusion |
| 9 | `SimilarityHead` | ArcFace-style cosine similarity scoring |
| 10 | `UncertaintyHead` | MC Dropout-based epistemic + aleatoric uncertainty |
| 11 | `UFMTransformer` | Full model assembling all components |

> **Note francaise** : *L'architecture est entierement implementee en PyTorch pur. La bibliotheque `timm` n'est requise que pour charger le backbone EfficientNet-B2 pre-entraine sur ImageNet.*

---

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 11.7 (for GPU training) or Apple Silicon (MPS backend)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-org/ufm-transformer.git
cd ufm-transformer/src
```

### Step 2: Create a Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Or using conda
conda create -n ufm python=3.10
conda activate ufm
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python main.py --help
```

---

## Dataset Setup

The project expects a **multimodal face + fingerprint dataset** organized as follows:

```
data/
  subject_001/
    face_001.jpg        # Face image (RGB)
    face_002.jpg
    fingerprint_001.png # Fingerprint image (grayscale)
    fingerprint_002.png
  subject_002/
    face_001.jpg
    fingerprint_001.png
  ...
```

### Recommended Dataset: Face/Fingerprint Multimodal Dataset (Kaggle)

1. **Download from Kaggle**:
   ```bash
   pip install kaggle
   # Set up your Kaggle API credentials (~/.kaggle/kaggle.json)
   kaggle datasets download -d your-dataset/face-fingerprint-multimodal
   unzip face-fingerprint-multimodal.zip -d ./data
   ```

2. **Organize the data**:
   ```bash
   # Ensure the folder structure matches the expected layout above
   # Each subject should have at least one face and one fingerprint image
   ```

3. **Verify**:
   ```bash
   python -c "
   from data_loader import get_dataloaders
   train, val, test = get_dataloaders('./data', batch_size=4)
   print(f'Train: {len(train.dataset)} | Val: {len(val.dataset)} | Test: {len(test.dataset)}')
   "
   ```

> **Note** : *Le jeu de donnees doit contenir des paires face + empreinte digitale pour chaque sujet. Le module `data_loader.py` gere automatiquement le decoupage train/val/test disjoint par sujet.*

---

## Quick Start

### Training (from scratch)

```bash
python main.py --mode train \
    --dataset_path ./data \
    --output_dir ./output \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --device auto
```

### Evaluation

```bash
python main.py --mode eval \
    --model_path ./output/checkpoints/best_model.pth \
    --dataset_path ./data \
    --output_dir ./output/eval \
    --split test
```

### Visualization

```bash
python main.py --mode visualize \
    --model_path ./output/checkpoints/best_model.pth \
    --dataset_path ./data \
    --output_dir ./output/viz \
    --viz_n_identities 10
```

---

## Detailed Usage

### Global Arguments (all modes)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | *required* | Operating mode: `train`, `eval`, or `visualize` |
| `--dataset_path` | *required* | Root directory containing the dataset |
| `--output_dir` | `./output` | Directory for checkpoints, logs, and results |
| `--model_path` | `None` | Path to a saved checkpoint (eval/visualize only) |
| `--device` | `auto` | Compute device: `auto`, `cuda`, `mps`, or `cpu` |
| `--seed` | `42` | Random seed for reproducibility |
| `--batch_size` | `32` | Mini-batch size |
| `--num_workers` | `4` | DataLoader worker processes |
| `--image_size` | `224` | Input image resolution (both modalities) |
| `--embed_dim` | `256` | Transformer embedding dimension |
| `--num_heads` | `8` | Number of attention heads |
| `--fp16` | `False` | Enable Automatic Mixed Precision (AMP) |
| `--debug` | `False` | Enable debug-level logging |

### 1. Training Mode

Training follows a **2-phase curriculum**:

- **Phase 1 (Unimodal Pre-training)**: Independently trains face and fingerprint encoders with ArcFace loss, freezing all fusion components.
- **Phase 2 (Joint Fine-tuning)**: Unfreezes all parameters, applies random modality dropout (30%), and optimizes the full composite loss with cosine annealing.

```bash
python main.py --mode train \
    --dataset_path ./data \
    --output_dir ./output \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --scheduler cosine \
    --warmup_epochs 5 \
    --eval_every 1 \
    --save_every 5 \
    --batch_size 32 \
    --embed_dim 256 \
    --num_heads 8 \
    --fp16
```

#### Training-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `50` | Total training epochs |
| `--lr` | `1e-4` | Initial learning rate |
| `--weight_decay` | `1e-5` | L2 regularization coefficient |
| `--scheduler` | `cosine` | LR scheduler: `cosine`, `step`, `plateau`, `none` |
| `--warmup_epochs` | `5` | Linear LR warmup epochs |
| `--eval_every` | `1` | Run validation every N epochs |
| `--save_every` | `5` | Save checkpoint every N epochs |
| `--resume` | `None` | Resume from a checkpoint path |

#### Resume Training

```bash
python main.py --mode train \
    --dataset_path ./data \
    --output_dir ./output \
    --resume ./output/checkpoints/checkpoint_epoch_025.pth \
    --epochs 100
```

### 2. Evaluation Mode

Evaluation computes standard biometric metrics and compares against 5 baseline models.

```bash
python main.py --mode eval \
    --model_path ./output/checkpoints/best_model.pth \
    --dataset_path ./data \
    --output_dir ./output/eval \
    --split test \
    --batch_size 64
```

#### Evaluation-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `test` | Dataset split to evaluate: `train`, `val`, `test` |

#### Metrics Computed

- **EER** (Equal Error Rate): Point where FAR = FRR
- **TAR@FAR**: True Acceptance Rate at specified False Acceptance Rates (1%, 0.1%, 0.01%)
- **AUC**: Area Under the ROC Curve
- **FNMR/FMR Curves**: False Non-Match vs. False Match Rates
- **DET Curve**: Detection Error Trade-off
- **Uncertainty Calibration**: Expected Calibration Error (ECE)

#### Baseline Models Compared

| Baseline | Description |
|----------|-------------|
| `ConcatenationFusion` | Feature concatenation + MLP classifier |
| `ScoreSumFusion` | Independent encoders with weighted score fusion |
| `DenseNetBiModal` | DenseNet processing concatenated modalities |
| `TransformerFusionNoUncertainty` | UFM without uncertainty head |
| `MDRLFusion` | Multi-level hierarchical representation learning |

#### LaTeX Table Export

Evaluation results are automatically exported as publication-ready LaTeX tables to:
```
./output/eval/results_table.tex
```

### 3. Visualization Mode

Generate attention maps and Grad-CAM overlays for model interpretability.

```bash
python main.py --mode visualize \
    --model_path ./output/checkpoints/best_model.pth \
    --dataset_path ./data \
    --output_dir ./output/viz \
    --viz_n_identities 10
```

#### Visualization-Specific Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--viz_n_identities` | `10` | Number of identities to visualize |
| `--viz_pairs_file` | `None` | JSON file with pre-selected verification pairs |

#### Generated Visualizations

1. **Attention Heatmaps**: Cross-modal attention weights between face and fingerprint patches
2. **Grad-CAM Overlays**: Class Activation Maps for face and fingerprint branches
3. **Cross-Attention Matrices**: Face regions vs. fingerprint minutiae attention
4. **Explainability Report**: Multi-page PDF combining all visualizations per identity

> **Note francaise** : *Les visualisations sont exportees en PDF haute resolution (300 DPI), adaptees a une insertion directe dans des articles scientifiques.*

---

## Project Structure

```
src/
|-- main.py              # Entry point: argument parsing + mode dispatch
|-- train.py             # Training pipeline (2-phase, composite loss)
|-- evaluate.py          # Evaluation metrics, baselines, visualizations, LaTeX export
|-- models.py            # UFM-Transformer architecture (11 classes)
|-- data_loader.py       # Dataset, augmentations, missing modality simulation
|-- utils.py             # Utilities (seed, device, checkpoints, logging, timing)
|-- visualize.py         # Attention maps, Grad-CAM, explainability reports
|-- requirements.txt     # Python dependencies
|-- README.md            # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates the full lifecycle: training, evaluation, and visualization. Parses CLI arguments and dispatches to the appropriate module. |
| `train.py` | Implements the 2-phase training procedure with `UFMLoss` (triplet + ArcFace + uncertainty), `ArcFaceMargin` layer, and `TrainConfig` dataclass for hyperparameters. |
| `evaluate.py` | Computes biometric metrics (EER, TAR@FAR, AUC), runs baseline comparisons, generates 7 types of publication-quality plots, and exports LaTeX tables. |
| `models.py` | Full UFM-Transformer implementation: modality-specific encoders, quality-aware cross-modal transformer, learnable missing-modality tokens, uncertainty decomposition via MC Dropout, and ArcFace scoring. |
| `data_loader.py` | `MultimodalDataset` with paired face/fingerprint loading, modality-specific augmentations (Gaussian noise, blur, rotation), `MissingModalitySimulator`, quality estimation, and subject-disjoint splitting. |
| `utils.py` | `AverageMeter`, `Timer`, `Logger`, `set_seed()`, `get_device()`, `save_checkpoint()`, `load_checkpoint()`, `count_parameters()`, `compute_flops()`, `print_system_info()`. |
| `visualize.py` | `extract_attention_maps()`, `visualize_cross_attention()`, `compute_gradcam_bimodal()`, `visualize_gradcam_bimodal()`, `generate_explainability_report()`, `plot_attention_heatmap()`, `GradCAMBimodal` class. |

---

## Hyperparameters

### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_dim` | `256` | Common embedding dimension after projection |
| `num_heads` | `8` | Attention heads in cross-modal transformer |
| `image_size` | `224` | Input resolution for both modalities |
| `face_encoder` | `efficientnet_b2` | Face backbone (timm) |
| `fp_encoder` | `Custom ResNet` | Fingerprint backbone channels: 64 -> 128 -> 256 -> 512 |
| `face_channels` | `1408` | EfficientNet-B2 output channels |
| `fp_channels` | `512` | Fingerprint encoder output channels |

### Training Configuration (`TrainConfig`)

| Parameter | Phase 1 | Phase 2 | Description |
|-----------|---------|---------|-------------|
| `epochs` | 50 | 100 | Training epochs per phase |
| `lr` | 1e-3 | 1e-4 | Peak learning rate |
| `lr_min` | - | 1e-6 | Minimum LR (cosine annealing) |
| `weight_decay` | 1e-4 | 1e-4 | L2 regularization |
| `max_grad_norm` | 1.0 | 1.0 | Gradient clipping |
| `warmup_epochs` | - | 5 | Linear LR warmup |
| `batch_size` | 64 | 64 | Mini-batch size |
| `modality_dropout` | 0.0 | 0.30 | Random modality masking probability |

### Loss Weights (`UFMLoss`)

| Component | Weight | Description |
|-----------|--------|-------------|
| Triplet loss | 1.0 | Hard negative mining, margin = 0.5 |
| ArcFace loss | 1.0 | Additive angular margin, m = 0.5, s = 30.0 |
| Uncertainty regularization | 0.1 | Uncertainty-aware weighting |

### Loss Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `triplet_margin` | 0.5 | Margin for semi-hard negative triplet mining |
| `arcface_margin` | 0.5 | Angular margin m (radians) |
| `arcface_scale` | 30.0 | Feature norm scale s |

---

## Expected Outputs

### Training Outputs

```
output/
|-- checkpoints/
|   |-- best_model.pth              # Best model (lowest validation EER)
|   |-- checkpoint_epoch_005.pth    # Periodic checkpoints
|   |-- checkpoint_epoch_010.pth
|   |-- ...
|-- logs/
|   |-- training.log                # Training log with per-epoch metrics
|   |-- config.json                 # Saved hyperparameters
|-- metrics.json                    # Per-epoch training/validation metrics
```

### Evaluation Outputs

```
output/eval/
|-- roc_curve.pdf                   # ROC curve with AUC annotation
|-- det_curve.pdf                   # Detection Error Trade-off curve
|-- eer_histogram.pdf               # EER distribution across folds
|-- confusion_matrix.pdf            # Confusion matrix heatmap
|-- calibration_plot.pdf            # Uncertainty calibration curve
|-- similarity_distribution.pdf     # Genuine vs. impostor similarity scores
|-- fnmr_fmr_curve.pdf              # FNMR/FMR vs. threshold
|-- results_table.tex               # LaTeX table for academic papers
|-- metrics.json                    # All numerical metrics
```

### Visualization Outputs

```
output/viz/
|-- attention_heatmap_subject_001.pdf     # Cross-modal attention heatmap
|-- gradcam_face_subject_001.pdf          # Grad-CAM overlay (face)
|-- gradcam_fingerprint_subject_001.pdf   # Grad-CAM overlay (fingerprint)
|-- cross_attention_matrix.pdf            # Aggregated attention matrix
|-- explainability_report.pdf             # Multi-page PDF with all plots
```

---

## Citation

If you use UFM-Transformer in your research, please cite:

```bibtex
@article{ufm_transformer2024,
  title={UFM-Transformer: Uncertainty-aware Fusion with Missing-modality Transformer for Multimodal Biometric Verification},
  author={UFM-Transformer Team},
  journal={arXiv preprint},
  year={2024}
}
```

### Related Works

This implementation builds upon:

- **ArcFace**: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," *CVPR*, 2019.
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," *ICML*, 2019.
- **Transformer**: Vaswani et al., "Attention Is All You Need," *NeurIPS*, 2017.
- **MC Dropout**: Gal & Ghahramani, "Dropout as a Bayesian Approximation," *ICML*, 2016.

---

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

> **Note** : *Ce projet est fourni a titre de recherche academique. Les poids pre-entraines d'EfficientNet-B2 sont distribues sous la licence Apache 2.0 via la bibliotheque `timm`.*

---

## Contact

For questions, bug reports, or contributions, please open an issue on GitHub or contact the UFM-Transformer Team.

---

*README generated for UFM-Transformer v1.0 -- Multimodal Biometric Verification Pipeline*
