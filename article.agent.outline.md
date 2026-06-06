# UFM-Transformer: Uncertainty-aware Multimodal Fusion for Robust Face and Fingerprint Verification

## Abstract (~200 words)
### Research Summary
#### Multimodal biometric fusion combining face and fingerprint suffers from representation misalignment under quality degradation, absence of principled missing-modality handling, and deterministic scores without confidence intervals
#### We propose UFM-Transformer, a cross-attention transformer with quality-modulated attention, learnable missing-modality tokens, Monte-Carlo uncertainty estimation, and bimodal Grad-CAM explainability
#### Evaluation on [DATASET] with 100+ subjects demonstrates \textcolor{red}{\textbf{XX\%}} EER reduction over best baseline and \textcolor{red}{\textbf{XX\%}} calibration error improvement
#### UFM-Transformer enables reliable identity verification with rejection option and interpretable decisions in open-set scenarios with variable-quality samples

## 1. Introduction (~1500 words, 1 figure: architecture overview)
### 1.1 Background and Motivation
#### 1.1.1 Biometric verification in open-set scenarios: face and fingerprint as complementary traits with orthogonal failure modes
#### 1.1.2 Deep learning revolution in unimodal biometrics: CNNs, Vision Transformers, and metric learning (ArcFace, AdaFace)
#### 1.1.3 The multimodal promise: why fusion should outperform unimodal systems, yet current approaches fall short
### 1.2 Research Gap and Problem Statement
#### 1.2.1 Four interconnected challenges: (1) representation misalignment under degradation, (2) no native missing-modality handling, (3) deterministic scores without uncertainty, (4) lack of bimodal explainability
#### 1.2.2 The "modality island" problem: existing works address each challenge in isolation
#### 1.2.3 Formal problem formulation: open-set verification with variable quality and partial modality availability
### 1.3 Contribution Statement
#### 1.3.1 UFM-Transformer: first unified framework addressing all four challenges simultaneously
#### 1.3.2 Quality-modulated cross-attention transformer with learnable missing-modality tokens
#### 1.3.3 Uncertainty-calibrated similarity scores via Monte-Carlo Dropout with aleatoric-epistemic decomposition
#### 1.3.4 Bimodal Grad-CAM and cross-attention visualization for interpretable decisions
#### 1.3.5 Extensive experimental validation: 5 baselines, 10 degradation conditions, calibration analysis
### 1.4 Paper Organization
#### 1.4.1 Section structure overview

## 2. Related Work (~2000 words, 1 comparison table)
### 2.1 Deep Multimodal Fusion Architectures
#### 2.1.1 Early fusion: feature concatenation and joint embedding learning — limitations under degradation
#### 2.1.2 Late fusion: score-level combination — loss of cross-modal information
#### 2.1.3 Hybrid fusion: multi-stage pipelines — computational cost and complexity
### 2.2 Attention and Transformer Mechanisms for Biometrics
#### 2.2.1 Self-attention for face recognition: TransFace, LVFace, ViT-GCT
#### 2.2.2 Cross-modal attention: quality-gated fusion, bottleneck attention
#### 2.2.3 Gap: no true inter-modal cross-attention for face+fingerprint verification
### 2.3 Robustness and Missing Modality Handling
#### 2.3.1 Quality-aware recognition: MagFace, QMagFace, CR-FIQA for face; NFIQ 2.0 for fingerprint
#### 2.3.2 Missing modality in medical imaging vs biometrics: 5-year lag
#### 2.3.3 Modality dropout, learnable tokens, hallucination networks — never applied to biometrics
### 2.4 Uncertainty Quantification in Biometric Verification
#### 2.4.1 Probabilistic face embeddings: PFE, DUL
#### 2.4.2 Monte-Carlo Dropout and Deep Ensembles for biometric confidence
#### 2.4.3 Gap: no uncertainty-aware MULTIMODAL biometric system
### 2.5 Explainability in Biometric Systems
#### 2.5.1 Grad-CAM for face recognition, attention maps for fingerprints
#### 2.5.2 Gap: no unified bimodal explanation (face regions + minutiae simultaneously)
### 2.6 Positioning Against Prior Work
#### 2.6.1 Comparison table: UFM-Transformer vs 6 closest works on 8 dimensions (fusion type, quality-aware, missing modality, uncertainty, explainability, params, EER)
#### 2.6.2 Closest work: Tiny Transformers (PeerJ 2026) — has quality gating but no missing modality, no uncertainty, no explainability
#### 2.6.3 Second closest: AuthFormer (arXiv 2024) — cross-attention but deterministic, no quality modulation

## 3. Proposed Method (~2500 words, 2 figures, 1 algorithm)
### 3.1 Problem Formulation
#### 3.1.1 Notation: face image $x_f$, fingerprint image $x_p$, identity label $y$, quality scores $q_f, q_p \in [0,1]$, missing mask $m \in \{0,1\}^2$
#### 3.1.2 Open-set verification objective: learn similarity function $s(x_f, x_p, x_f', x_p')$ with calibrated confidence interval
### 3.2 Architecture Overview
#### 3.2.1 High-level pipeline: modality-specific encoders → projectors → quality-modulated cross-attention transformer → similarity + uncertainty heads
#### 3.2.2 Figure: complete architecture diagram with dimension flow
### 3.3 Modality-Specific Encoders
#### 3.3.1 Face encoder: EfficientNet-B2 from timm, ImageNet pretraining, feature map $f_f \in \mathbb{R}^{B \times 1408 \times 7 \times 7}$
#### 3.3.2 Fingerprint encoder: custom ResNet-style CNN with residual blocks, feature map $f_p \in \mathbb{R}^{B \times 512 \times 7 \times 7}$
#### 3.3.3 Quality estimator: lightweight 4-block CNN → quality score $q \in [0,1]$ per modality
### 3.4 Common Projection and Missing-Modality Handling
#### 3.4.1 L2-normalized projectors: face and fingerprint features → 256-dim unit hypersphere
#### 3.4.2 Learnable missing-modality token: $\mathbf{t}_{miss} \in \mathbb{R}^{256}$ as learnable parameter
#### 3.4.3 Modality replacement: if $m_i = 0$, replace projected features with $\mathbf{t}_{miss}$ + positional encoding
### 3.5 Quality-Modulated Cross-Attention Transformer
#### 3.5.1 Two-stream cross-attention: face queries → fingerprint keys/values, and vice versa
#### 3.5.2 Quality modulation: attention scores scaled by quality product $q_f \cdot q_p$ before softmax
#### 3.5.3 Missing-aware masking: zero attention from/to missing modalities via boolean mask
#### 3.5.4 Architecture: 4 layers, 8 heads, 256-dim, residual connections, layer normalization
#### 3.5.5 Algorithm: forward pass pseudocode
### 3.6 Similarity and Uncertainty Heads
#### 3.6.1 Similarity head: cosine similarity with ArcFace additive angular margin ($m=0.5$, $s=30$)
#### 3.6.2 Uncertainty head: 5 forward passes with dropout enabled, variance decomposition
#### 3.6.3 Aleatoric vs epistemic: data noise vs model uncertainty decomposition
#### 3.6.4 Confidence interval: similarity score $\pm \kappa \cdot \sigma_{total}$ for rejection option
### 3.7 Bimodal Explainability
#### 3.7.1 Cross-attention visualization: which face patches attend to which fingerprint regions
#### 3.7.2 Grad-CAM bimodal: gradient-weighted class activation for both modalities
#### 3.7.3 Attention heatmaps for failure case analysis
### 3.8 Training Strategy
#### 3.8.1 Composite loss: $\mathcal{L} = \mathcal{L}_{triplet} + \lambda_1 \mathcal{L}_{arcface} + \lambda_2 \mathcal{L}_{uncertainty}$
#### 3.8.2 Phase 1 (50 epochs): unimodal pretraining — encoders + projectors with ArcFace loss
#### 3.8.3 Phase 2 (100 epochs): joint fine-tuning — full model with modality dropout (30%) and composite loss
#### 3.8.4 Hyperparameters: AdamW, lr=1e-4 with cosine annealing, batch size 64, gradient clipping

## 4. Experiments (~2500 words, 4 tables, 3 figures)
### 4.1 Experimental Setup
#### 4.1.1 Dataset: Multimodal Biometrics Dataset (Face, Fingerprint, Iris, Signature) from Kaggle — 100+ subjects, 5+ samples per modality
#### 4.1.2 Preprocessing: resize to 224×224, normalization, subject-disjoint split 70/15/15
#### 4.1.3 Augmentations: blur (σ=2,4), noise (σ=0.05), occlusion (30% face), minutiae dropout (50%)
#### 4.1.4 Implementation: PyTorch 2.0, CUDA, mixed precision training
#### 4.1.5 Hardware: NVIDIA GPU with XX GB VRAM
### 4.2 Baselines
#### 4.2.1 ConcatenationFusion: feature concat + MLP
#### 4.2.2 ScoreSumFusion: independent encoders + quality-weighted score sum
#### 4.2.3 DenseNetBiModal: DenseNet on concatenated 6-channel input
#### 4.2.4 TransformerFusionNoUncertainty: same transformer fusion without uncertainty head
#### 4.2.5 MDRLFusion: multi-deep representation learning with hierarchical fusion
### 4.3 Main Results
#### 4.3.1 Verification metrics comparison: EER, TAR@FAR=0.1%, TAR@FAR=1%, AUC (Table 1)
#### 4.3.2 UFM-Transformer achieves \textcolor{red}{\textbf{XX\%}} EER vs \textcolor{red}{\textbf{XX\%}} best baseline
#### 4.3.3 ROC and DET curves (Figure)
### 4.4 Robustness Analysis
#### 4.4.1 Degradation conditions: clean, face-blur, face-noise, face-occlusion, fp-blur, fp-noise, fp-minutiae-dropout, both-degraded, face-only, fp-only (Table 2)
#### 4.4.2 UFM maintains EER within XX% of clean under severe degradation
#### 4.4.3 Score distribution violin plots per condition
### 4.5 Uncertainty Calibration
#### 4.5.1 Reliability diagram: confidence vs accuracy (Figure)
#### 4.5.2 ECE and MCE values for UFM and baselines (Table 3)
#### 4.5.3 Uncertainty histograms: correct vs incorrect predictions
#### 4.5.4 Rejection option: performance vs fraction rejected (Figure)
### 4.6 Ablation Study
#### 4.6.1 Component ablation: no quality modulation, no missing token, no uncertainty, no cross-attention (Table 4)
#### 4.6.2 Quality modulation contribution: XX% EER improvement
#### 4.6.3 Missing-modality token contribution: graceful degradation analysis
### 4.7 Explainability Results
#### 4.7.1 Cross-attention heatmaps for 10 test identities
#### 4.7.2 Grad-CAM overlays showing facial regions and minutiae importance
#### 4.7.3 Failure case analysis: when and why the model fails

## 5. Conclusion (~400 words)
### 5.1 Summary of Contributions
#### 5.1.1 UFM-Transformer: unified uncertainty-aware multimodal fusion
#### 5.1.2 Key results: EER \textcolor{red}{\textbf{XX\%}}, TAR@0.1\%FAR \textcolor{red}{\textbf{XX\%}}, ECE \textcolor{red}{\textbf{XX\%}}
### 5.2 Limitations and Future Work
#### 5.2.1 Dataset scale: extend to 10K+ subjects for large-scale evaluation
#### 5.2.2 Additional modalities: iris, voice, palmprint integration
#### 5.2.3 Real-time optimization: knowledge distillation for edge deployment

# References
## article_outline_references_raw.md
- **Type**: Citation collection
- **Path**: /mnt/agents/output/article_outline_references_raw.md
