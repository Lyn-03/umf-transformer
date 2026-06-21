# Cross-Verification Report: Deep Learning for Multimodal Biometrics

**Document Type:** Cross-Dimension Synthesis and Verification
**Dimensions Covered:** 10 (Early Fusion, Late Fusion, Hybrid Fusion, Transformers, Cross-Modal Attention, Face Robustness, Fingerprint Robustness, Missing Modality, Uncertainty, Explainability)
**Date Compiled:** 2025-07-01
**Sources Cross-Referenced:** 150+ papers across all 10 dimension files

---

## High Confidence Findings (confirmed by >=2 dimensions from independent sources)

### HC-1: Cross-Modal Attention Outperforms Simple Concatenation and Score-Level Fusion
- **Confirmed by:** Dim01 (Early Fusion), Dim04 (Transformers), Dim05 (Cross-Modal Attention), Dim03 (Hybrid Fusion)
- **Evidence:** Gnanapraveen et al. (Odyssey 2024) [Dim05] demonstrate cross-attention achieves EER 2.387 vs. self-attention 2.412 vs. feature concatenation 2.489 vs. score-level fusion 2.521. Tiong et al. (CVPR 2024) [Dim05] confirm MFA-ViT sets SOTA on face-periocular recognition. Dim03 identifies transformer-based cross-attention as a dominant trend in hybrid architectures.
- **Cross-verification status:** **VERIFIED** — four independent dimensions converge on this conclusion across audio-visual, face-periocular, and multimodal biometric benchmarks.

### HC-2: Quality-Aware Fusion Is Essential for Real-World Deployment
- **Confirmed by:** Dim02 (Late Fusion), Dim05 (Cross-Modal Attention), Dim06 (Face Robustness), Dim07 (Fingerprint Robustness), Dim09 (Uncertainty)
- **Evidence:** Soleymani et al. (IEEE TBIOM 2021) [Dim05] show >30% TAR improvement at FAR=10^-4 with weakly-supervised quality scores. QMagFace (WACV 2023) [Dim06] achieves 98.74% on CFP-FP via quality-aware scoring. Priesnitz et al. (BIOSIG 2025) [Dim07] demonstrate deep learning quality assessment outperforms NFIQ 2.3 by 12.29%. NIJ research [Dim07] validates Bayesian score fusion incorporating quality. Uncertainty quantification methods (PFE, DUL) [Dim09] naturally correlate with quality.
- **Cross-verification status:** **VERIFIED** — five dimensions independently converge that quality-aware fusion is operationally necessary and technically validated.

### HC-3: Transformer Architectures Are Replacing CNNs for Biometric Recognition
- **Confirmed by:** Dim04 (Transformers), Dim05 (Cross-Modal Attention), Dim03 (Hybrid Fusion), Dim01 (Early Fusion)
- **Evidence:** LVFace (2025) [Dim04] achieves 1st place on MFR-Ongoing with ViT-L. AFR-Net (IEEE T-BIOM 2024) [Dim04] achieves 99.78% rank-1 on NIST SD14. AuthFormer (2024) [Dim05] achieves 99.73% accuracy with 2 encoder layers. Tiny Transformers (2026) [Dim04/05] achieve <50ms inference. Dim03 identifies transformer-based cross-modal attention as a dominant trend.
- **Cross-verification status:** **VERIFIED** — four dimensions confirm transformer architectures are becoming dominant, though CNNs remain competitive with proper training (TopoFR with ResNet outperforms TransFace in some cases [Dim04]).

### HC-4: Missing Modality Handling Is Critically Under-Addressed in Biometrics
- **Confirmed by:** Dim08 (Missing Modality), Dim02 (Late Fusion), Dim03 (Hybrid Fusion), Dim05 (Cross-Modal Attention)
- **Evidence:** Dim08 survey finds that "multimodal biometric face+fingerprint fusion papers overwhelmingly assume all modalities are present." Only 3 of 17 surveyed methods are biometric-specific, and ALL use non-deep-learning approaches (score-level fallback or decision-level fusion). Dim03 identifies missing modality handling for face+fingerprint as "the most critical gap." Dim08 confirms ModDrop (Neverova et al., T-PAMI 2016) is foundational but underutilized in biometrics. Ma et al. (CVPR 2022) [Dim08] found that multimodal transformers are NOT naturally robust to missing modalities — they can perform worse than unimodal models.
- **Cross-verification status:** **VERIFIED** — four dimensions independently flag this as a critical gap with strong consensus.

### HC-5: Feature-Level Fusion Is Superior to Score-Level When Sufficient Data Exists
- **Confirmed by:** Dim01 (Early Fusion), Dim02 (Late Fusion), Dim03 (Hybrid Fusion), Dim05 (Cross-Modal Attention)
- **Evidence:** Liang et al. (ACM Computing Surveys 2024) [Dim02] characterize decision-level fusion as having "obvious drawbacks" with performance "limited by one modality." ACM survey [Dim02] confirms feature-level fusion "consistently outperforms score-level fusion when sufficient training data is available." Alay & Al-Baity (2020) [Dim01/02] achieve 99.39% with feature-level vs. 100% score-level, but the score-level superiority is attributed to well-calibrated classifiers on high-quality data, not inherent advantage. Hybrid fusion (combining both) consistently outperforms either alone (99.79% hybrid vs. 99.37% score-only [Dim03]).
- **Cross-verification status:** **VERIFIED with nuance** — the superiority depends on data quality and classifier calibration; hybrid fusion is the consensus best approach.

### HC-6: Probabilistic Embeddings (PFE, DUL) Provide Calibrated Uncertainty for Face Recognition
- **Confirmed by:** Dim09 (Uncertainty), Dim06 (Face Robustness)
- **Evidence:** Shi & Jain (ICCV 2019) [Dim09] pioneer PFE representing faces as Gaussian distributions N(z; mu, sigma^2). Chang et al. (CVPR 2020) [Dim09] extend with DUL achieving 90.23% TPR@FPR=0.001% on IJB-C vs. 83.06% baseline. The learned variance correlates with image quality degradation [Dim06/Dim09]. HolUE (2024) [Dim09] combines gallery-aware uncertainty with sample quality. LAM (NeurIPS 2023) [Dim09] achieves SOTA OOD detection with lower ECE.
- **Cross-verification status:** **VERIFIED** — multiple independent works confirm distributional embeddings improve both accuracy and provide calibrated uncertainty.

### HC-7: Self-Supervised ViTs Learn Minutiae-Free Fingerprint Representations
- **Confirmed by:** Dim04 (Transformers), Dim07 (Fingerprint Robustness), Dim10 (Explainability)
- **Evidence:** Gazi et al. (2026) [Dim04/07] demonstrate DINOv2-Base achieves 5.56% EER vs. VeriFinger 26.90% on multi-sensor data, with attention maps spontaneously aligning to minutiae regions (IoU 0.41+-0.07) [Dim10]. AFR-Net (Grosz & Jain, IEEE T-BIOM 2024) [Dim04] combines CNN+ViT achieving 99.78% rank-1 on NIST SD14. Medium-capacity models (86M params) outperform 1.1B parameter giants [Dim07].
- **Cross-verification status:** **VERIFIED** — three independent dimensions confirm the viability of self-supervised transformer-based fingerprint recognition.

### HC-8: Explainability Is Gaining Critical Importance for Biometric AI Regulation
- **Confirmed by:** Dim10 (Explainability), Dim04 (Transformers), Dim09 (Uncertainty)
- **Evidence:** Neto et al. survey (2022) [Dim10] establishes "xAI Ladder" taxonomy. EU AI Act and similar regulations require explainability for high-risk AI systems including biometrics [Dim10]. CorrRISE (WACV 2024) [Dim10] outperforms Grad-CAM++ in face verification benchmarks. SHAP/LIME integration into multimodal biometric pipelines is emerging [Dim10]. Self-supervised ViT attention-minutiae alignment provides forensically meaningful explanations [Dim10].
- **Cross-verification status:** **VERIFIED** — regulatory drivers and technical progress converge across multiple dimensions.

### HC-9: Modality Dropout (ModDrop) Is the Foundational Technique for Missing Modality Robustness
- **Confirmed by:** Dim08 (Missing Modality), Dim03 (Hybrid Fusion)
- **Evidence:** Neverova et al. (T-PAMI 2016) [Dim08] introduced ModDrop, which remains the most widely adopted training-time technique. Random dropping of modality channels forces networks to learn cross-modal correlations while preserving modality-specific representations. Woo et al. (AAAI 2023) [Dim08] confirm transformer-based fusion is more robust to missing modalities than summation or concatenation. Shen et al. (2025) [Dim08] extend to Multimodal Adaptive Dropout (MAD) for biometric modality balancing. CMPTs (Reza et al., 2025) [Dim08] improve upon ModDrop with learnable proxy tokens.
- **Cross-verification status:** **VERIFIED** — confirmed as foundational across general ML and biometric-specific contexts.

### HC-10: Bilinear/Multiplicative Fusion Outperforms Linear Concatenation
- **Confirmed by:** Dim01 (Early Fusion), Dim05 (Cross-Modal Attention)
- **Evidence:** Soleymani et al. (ICIP 2018) [Dim01] demonstrate Generalized Compact Bilinear (GCB) fusion achieves SOTA on CMU Multi-PIE, BioCop, and BIOMDATA. Talreja et al. (2019) [Dim01] achieve EER of 0.84% with bilinear fusion vs. higher EER with linear concatenation. Dim05 confirms compact bilinear pooling via tensor sketch projection achieves comparable performance to full bilinear pooling with dramatically fewer parameters.
- **Cross-verification status:** **VERIFIED** — multiplicative interactions between modality features consistently outperform additive approaches.

### HC-11: Hybrid Fusion (Feature + Score) Achieves the Highest Reported Accuracies
- **Confirmed by:** Dim03 (Hybrid Fusion), Dim01 (Early Fusion), Dim02 (Late Fusion)
- **Evidence:** Aswin et al. (2025) [Dim03] achieve 99.79% accuracy with 70% score/30% feature hybrid on NUPT-FPV vs. 99.37% score-only. Al-Askar et al. (2025) [Dim01/03] confirm hybrid fusion achieves lowest EER of 0.18% compared to 0.28% for score-level. Multiple 2023-2025 studies [Dim03] combine feature-level and score-level fusion rather than using either in isolation.
- **Cross-verification status:** **VERIFIED** — hybrid fusion is the dominant paradigm for achieving top accuracy.

### HC-12: Fixed-Length Deep Representations (DeepPrint, AFR-Net) Are Inherently Robust to Degraded Fingerprints
- **Confirmed by:** Dim07 (Fingerprint Robustness), Dim04 (Transformers)
- **Evidence:** DeepPrint (Engelsma et al., T-PAMI 2021) [Dim07] learns 200-byte representations that avoid graph matching and perform reliably when minutiae extraction fails. AFR-Net (Grosz & Jain, IEEE T-BIOM 2024) [Dim04/07] combines CNN+ViT with realignment for low-certainty situations, achieving 99.78% rank-1 on NIST SD14. PFVNet (He et al., IEEE T-IFS 2022) [Dim07] learns pose-invariant embeddings from partial prints. JIPNet (Guan et al., 2025) [Dim07] jointly optimizes identity and pose alignment.
- **Cross-verification status:** **VERIFIED** — multiple architectures confirm fixed-length embeddings outperform minutiae-based approaches under degradation.

---

## Medium Confidence Findings (1 authoritative source or limited cross-dimension confirmation)

### MC-1: Phase-Only Cross-Attention (POC-ViT) for Illumination-Invariant Authentication
- **Source:** Dim05 (Cross-Modal Attention) — Sharma et al., arXiv:2412.19160, 2024/2025
- **Finding:** Phase correlation in the frequency domain enables cross-attention robust to illumination/resolution variations, achieving 98.8% accuracy on forehead vein + periocular recognition.
- **Confidence:** Medium — single source, preprint status at time of search, limited evaluation scope.

### MC-2: AuthFormer Achieves 99.73% with Only 2 Encoder Layers
- **Source:** Dim03 (Hybrid Fusion), Dim05 (Cross-Modal Attention) — Yang et al., arXiv:2411.05395, 2024
- **Finding:** Adaptive multimodal biometric authentication transformer with cross-attention + GRN achieves 99.73% accuracy on face+fingerprint+voice with 2 encoder layers.
- **Confidence:** Medium — results confirmed in two dimensions but paper is arXiv preprint; dataset (LUTBIO) has limited public availability.

### MC-3: Graph Attention Networks Provide Inherently Explainable Multimodal Fusion
- **Source:** Dim10 (Explainability) — Harini et al., 2025-2026
- **Finding:** GAT treats modalities as nodes in a fully connected graph with learned inter-modal attention weights providing inherent explainability.
- **Confidence:** Medium — student thesis work, small dataset (75 subjects), limited peer review.

### MC-4: Spiking Neural Networks Achieve 1/100th Energy Consumption
- **Source:** Dim03 (Hybrid Fusion) — Shen et al., 2025
- **Finding:** SNN-based multimodal biometric fusion achieves 95.31% accuracy on CASIA with only 1.32 mJ energy consumption (~1/100th of ANN-based methods).
- **Confidence:** Medium — novel venue (Journal of Advanced Research in CS and Engineering), needs independent verification.

### MC-5: Cross-Stitch Networks Have Not Been Systematically Applied to Face+Fingerprint
- **Source:** Dim01 (Early Fusion) — Misra et al. (CVPR 2016), Ruder et al. (AAAI 2019)
- **Finding:** Despite theoretical promise for adaptive feature sharing, cross-stitch/sluice networks have not been directly applied to face+fingerprint biometric fusion with deep learning.
- **Confidence:** Medium — verified through literature search absence, but absence of evidence is not evidence of absence.

### MC-6: Moderate-Capacity Transformers (86M params) Outperform Giants for Fingerprint Data
- **Source:** Dim04 (Transformers), Dim07 (Fingerprint Robustness) — Gazi et al., 2026
- **Finding:** DINOv2-Base (86M) achieves 5.56% EER while Giant (1.1B) degrades to 5.80% on heterogeneous fingerprint data — opposite to face recognition where scaling helps.
- **Confidence:** Medium — confirmed in two dimensions but from the same paper; challenges the "bigger is better" assumption for limited biometric data.

### MC-7: Quality-Guided Mixture of Score-Fusion Experts (QME) Handles Missing Modalities
- **Source:** Dim03 (Hybrid Fusion) — Zhu et al., ICCV 2025
- **Finding:** QME uses MoE with quality-guided dynamic weighting to adapt to sensor noise, occlusions, and missing modalities for whole-body biometrics.
- **Confidence:** Medium — high-venue acceptance (ICCV 2025), but evaluated on face+gait+body rather than face+fingerprint specifically.

### MC-8: Gabor-Enhanced Attention Networks Achieve 99.49% at 0.85 GFLOPs
- **Source:** Dim05 (Cross-Modal Attention) — Than & Nguyen, JRC 2025
- **Finding:** Learnable Gabor filters combined with dynamic attention-driven fusion achieve 99.49% accuracy and 0.35% EER with only 10.6M parameters.
- **Confidence:** Medium — newer venue (Journal of Robotics and Control), results need independent verification [CITATION A VERIFIER].

### MC-9: Synthetic Degradation (WebFace-OCC) Significantly Improves Occlusion Robustness
- **Source:** Dim06 (Face Robustness) — Huang et al., ICASSP 2021
- **Finding:** Training on 804,704 synthetically occluded images of 10,575 subjects significantly improves face recognition under occlusion.
- **Confidence:** Medium — high-quality source but synthetic occlusion may not capture all real-world patterns.

### MC-10: Multimodal Prompt Tuning (MPT) Enables Parameter-Efficient Modality Alignment
- **Source:** Dim05 (Cross-Modal Attention) — Tiong et al., CVPR 2024
- **Finding:** Multimodal prompt tuning avoids modality-specific fine-tuning by aligning modalities within a shared ViT backbone.
- **Confidence:** Medium — demonstrated for face+periocular, extensibility to face+fingerprint is plausible but not tested.

---

## Low Confidence Findings (weak sourcing or preliminary results)

### LC-1: Quaternion-Based Fusion in RKHS Achieves AUC 0.9813
- **Source:** Dim01 (Early Fusion) — Safavipour et al., Computational Intelligence and Neuroscience, 2023
- **Finding:** Feature-level fusion in Reproducing Kernel Hilbert Space using quaternion-based parallel fusion achieves AUC of 0.9813 for iris and 0.9123 for fingerprint.
- **Confidence:** Low — lower-tier venue, limited experimental validation, complex methodology without independent replication.

### LC-2: CNN Concatenation-Based Early Fusion Enhances Discriminative Power
- **Source:** Dim01 (Early Fusion) — Springer Journal, 2025
- **Finding:** Simple concatenation of flattened CNN outputs from face and fingerprint branches prior to classification enhances discriminative power.
- **Confidence:** Low — general claim without novel contribution, confirmed by many earlier works.

### LC-3: Face Restoration (GFP-GAN, DifFace) Improves Degraded Recognition
- **Source:** Dim06 (Face Robustness)
- **Finding:** Face restoration methods can serve as preprocessing pipelines, with DifFace achieving best FID but requiring 11.38s vs. GFP-GAN at 0.06s.
- **Confidence:** Low for operational deployment — identity preservation during restoration remains a key challenge; restoration may introduce hallucinated features.

### LC-4: Whale Optimization Algorithm Achieves 99.6% Accuracy
- **Source:** Dim01 (Early Fusion) — Kumar et al., IJACSA, 2021
- **Finding:** Whale optimization algorithm with minutiae features achieves 99.6% accuracy with feature-level fusion using PatternNet+SVM.
- **Confidence:** Low — lower-tier venue (IJACSA), limited comparison with modern deep learning approaches.

### LC-5: Tiny Transformers Achieve <50ms Inference with <3M Parameters
- **Source:** Dim04 (Transformers), Dim05 (Cross-Modal Attention)
- **Finding:** Tiny ViT + Swin Transformer with quality-gated fusion achieves inference latency of <50ms with <3M trainable parameters.
- **Confidence:** Low-Medium — PeerJ venue is credible but not top-tier; the accuracy numbers on biometric benchmarks are not fully reported.

---

## Conflict Zones

### CZ-1: Feature-Level vs. Score-Level Fusion Superiority
- **Dimension 01 (Early Fusion):** Reports Alay & Al-Baity achieve 99.39% feature-level vs. 100% score-level on SDUMLA-HMT.
- **Dimension 02 (Late Fusion):** Reports the same study but emphasizes score-level uses "fixed and not trained rules."
- **Dimension 03 (Hybrid Fusion):** Resolves the conflict — hybrid fusion (99.79%) consistently outperforms both pure score-level (99.37%) and pure feature-level.
- **Resolution:** The apparent contradiction is resolved by recognizing that (a) score-level can win when individual classifiers are well-calibrated on high-quality data, (b) feature-level wins when cross-modal interactions are important, and (c) hybrid fusion combines both strengths. The superiority is dataset-dependent.

### CZ-2: Swin Transformer vs. ViT for Fingerprint Recognition
- **Dimension 04 (Transformers):** Cewers & Svensson thesis (2024) finds "the Swin Transformer underperformed and incurred higher inference costs, making it less suitable for deployment."
- **Dimension 04 (same):** Garg et al. (2025) successfully use Swin Transformer for multimodal biometric recognition achieving 99% accuracy.
- **Resolution:** The difference likely stems from training data scale, pretraining strategy, and task specifics (fingerprint-only vs. multimodal). For fingerprint-only matching, ViT may be preferable; for multimodal fusion with adequate data, Swin Transformer is viable.

### CZ-3: Minutiae-Free vs. Minutiae-Guided Fingerprint Recognition
- **Dimension 04 (Transformers):** Grosz et al. (2022) argue minutiae-guided ViT significantly improves latent recognition.
- **Dimension 04/07:** Gazi et al. (2026) demonstrate fully minutiae-free self-supervised ViTs outperform commercial systems on cross-sensor benchmarks.
- **Dimension 07:** Grosz & Jain (2023) propose a middle ground: fusing both local (minutiae) and global (embedding) features yields best results (LFR-Net).
- **Resolution:** Minutiae-free is preferable for generalization and simplicity; minutiae-guided for forensic/latent scenarios where minutiae annotations are available; hybrid (LFR-Net) for maximum accuracy when both are available.

### CZ-4: Training Strategy vs. Architecture as Key Differentiator
- **Dimension 04 (Transformers):** TransFace, LVFace, TopoFR all focus on training strategies rather than architectural innovation.
- **Dimension 04 (same):** TopoFR with ResNet50 outperforms TransFace (ViT-based) on some benchmarks, suggesting training matters more than architecture.
- **Dimension 06:** AdaFace, MagFace, ElasticFace all focus on loss function design rather than architecture.
- **Resolution:** For face recognition, training strategy (loss design, data augmentation, curriculum) appears to matter more than backbone choice. For fingerprint recognition, architecture choice (CNN+ViT hybrid) matters more due to limited training data.

### CZ-5: Feature Norm as Quality Proxy — Sufficient or Not?
- **Dimension 06 (Face Robustness):** AdaFace and MagFace use feature norm (embedding magnitude) as quality proxy.
- **Dimension 06 (same):** SlackedFace argues embedding norm alone is insufficient — unrecognizable hard examples may have large norms.
- **Resolution:** Simple norm-based quality is practical but may fail in extreme degradation. P-Norm (combining norm and proximity) or explicit quality networks provide more reliable estimates at modest computational cost.

### CZ-6: Generation-Based vs. Adaptation-Based Missing Modality Approaches
- **Dimension 08 (Missing Modality):** Generation-based methods (ActionMAE, GAN-based imputation) claim better performance but require more computation and may introduce artifacts.
- **Dimension 08 (same):** Adaptation-based methods (prompt tuning, LoRA, CMPT) are more efficient but may not fully recover performance.
- **Resolution:** The field is shifting from generation to adaptation (Trend 1 in Dim08). For biometrics, adaptation is preferable because reconstructing a missing fingerprint from a face is extremely difficult due to weak cross-modal correlation.

### CZ-7: Gradient-Based vs. Perturbation-Based Explainability
- **Dimension 10 (Explainability):** CorrRISE (perturbation-based) consistently outperforms Grad-CAM/Grad-CAM++ (gradient-based) in face verification benchmarks (Deletion: 23.29% vs. 50.65%).
- **Dimension 10 (same):** xSSAB (gradient-based) is more computationally efficient and provides training-free explanations.
- **Resolution:** Perturbation-based methods provide higher-quality explanations but at significant computational cost; gradient-based methods are more practical for real-time deployment. The choice depends on the use case (forensic analysis vs. real-time feedback).

### CZ-8: CLS Token vs. Global Context Token for Face Recognition
- **Dimension 04 (Transformers):** ViT-GCT (ICLR 2026) demonstrates that the standard CLS token is suboptimal, with GCT consistently outperforming CLS-only by +1.0% on IJB-C at FAR=1e-5.
- **Dimension 04 (same):** Standard ViT design universally uses CLS token.
- **Resolution:** GCT is a recent finding (submitted ICLR 2026). If validated, it challenges the standard ViT design choice for face recognition and potentially for biometric transformers more broadly.

---

## Gap Analysis Summary

### Tier 1: Critical Gaps (identified by >=3 dimensions)

#### GAP-C1: **No Face+Fingerprint Cross-Modal Attention with Native Missing Modality Handling**
- **Identified in:** Dim01, Dim03, Dim04, Dim05, Dim08
- **Description:** Despite extensive work on cross-modal attention for face+periocular, face+iris, and face+voice, no published work implements true inter-modal cross-attention where face patches and fingerprint patches directly attend to each other within a shared transformer space for face+fingerprint biometric verification. More critically, NONE of the cross-attention approaches natively handle missing modalities — they all assume both modalities are present at inference. This is the single most important gap.

#### GAP-C2: **No Uncertainty-Calibrated Multimodal Biometric Fusion**
- **Identified in:** Dim09, Dim02, Dim03, Dim07
- **Description:** Despite rich literature on uncertainty quantification for face recognition (PFE, DUL, HolUE, LAM) and for fingerprint enhancement (MU-GAN, DU-GAN), and despite extensive work on multimodal biometric fusion, the intersection of these two areas is essentially empty. No comprehensive framework propagates uncertainty from each modality through the fusion process to produce calibrated multimodal confidence scores with "don't know" rejection options.

#### GAP-C3: **No Bimodal Visual Explainability (Face Regions + Fingerprint Minutiae Simultaneously)**
- **Identified in:** Dim10, Dim05, Dim03
- **Description:** No published work provides unified visual explanations simultaneously highlighting which facial regions AND which fingerprint minutiae points contributed to a bimodal (face+fingerprint) biometric decision. Current multimodal explainability quantifies modality-level contributions via SHAP but does not generate fused visual saliency maps overlaying both face and fingerprint explanations.

#### GAP-C4: **No Large-Scale Public Dataset for Paired Face+Fingerprint**
- **Identified in:** Dim01, Dim07, Dim08
- **Description:** While NUPT-FPV (fingerprint+finger vein, 140 subjects), SDUMLA-HMT (face+iris+finger vein, 106 subjects), and BioCop (face+iris+fingerprint, 1000+ subjects) exist, there is no large-scale public dataset specifically designed for face+fingerprint paired biometric evaluation with quality annotations, controlled degradation, and missing modality protocols.

### Tier 2: Important Gaps (identified by 2 dimensions)

#### GAP-I1: **No Joint Quality Estimation Considering Cross-Modality Correlations**
- **Identified in:** Dim06, Dim07
- **Description:** Quality-aware fusion frameworks treat face quality and fingerprint quality as independent. No work models how face quality and fingerprint quality may be correlated (e.g., both degraded in poor lighting) or complementary.

#### GAP-I2: **No Standardized Benchmarks for Quality-Aware Multimodal Fusion**
- **Identified in:** Dim06, Dim09
- **Description:** NIST FATE evaluates FIQA in isolation; no benchmark exists for evaluating quality-guided face+fingerprint fusion under controlled, graded quality degradation.

#### GAP-I3: **Limited Cross-Dataset Generalization Studies**
- **Identified in:** Dim03, Dim07
- **Description:** Most studies evaluate on single datasets. Evaluation of hybrid fusion architectures across multiple datasets with domain shift (contact vs. contactless fingerprints, visible vs. NIR face) is limited.

#### GAP-I4: **No Intrinsically Interpretable Multimodal Fusion Architectures**
- **Identified in:** Dim10, Dim05
- **Description:** Current intrinsically interpretable approaches (ProtoSiamese, MaskSiamese, xCos) only address unimodal face recognition. No work has extended these concepts to multimodal biometric fusion.

#### GAP-I5: **Deep Ensembles for Biometric Recognition Uncertainty Are Under-Studied**
- **Identified in:** Dim09, Dim06
- **Description:** Despite being a gold-standard UQ method in general ML, deep ensembles have seen limited application in biometric recognition. Most works use single-model approaches (PFE, DUL) or MC-Dropout.

### Tier 3: Emerging Gaps (identified by 1 dimension)

#### GAP-E1: **Federated Hybrid Fusion for Biometrics** (Dim03)
#### GAP-E2: **Neural Architecture Search for Biometric Fusion** (Dim03)
#### GAP-E3: **Adversarial Robustness of Hybrid Fusion Architectures** (Dim03)
#### GAP-E4: **Dynamic Cascading Based on Real-Time Quality Assessment** (Dim03)
#### GAP-E5: **Conformal Prediction for Multimodal Biometric Fusion** (Dim09)
#### GAP-E6: **Real-Time Explainability for Multimodal Biometric Systems** (Dim10)
#### GAP-E7: **Explainable Deep Learning for Latent Fingerprint Matching** (Dim10)
#### GAP-E8: **Security Implications of Missing Modality Compensation** (Dim08)

---

## Reference Collection

### Foundational/Seminal Papers

1. Jain, A.K.; Nandakumar, K.; Ross, A. "Score Normalization in Multimodal Biometric Systems." *Pattern Recognition*, 2005, 38(12), 2270-2285. DOI: 10.1016/j.patcog.2005.06.016
2. Dass, S.C.; Nandakumar, K.; Jain, A.K. "A Principled Approach to Score Level Fusion in Multimodal Biometric Systems." *AVBPA 2005*, LNCS 3546, Springer, pp. 1049-1058. DOI: 10.1007/11527923_109
3. Nandakumar, K.; Chen, Y.; Dass, S.C.; Jain, A.K. "Likelihood Ratio-Based Biometric Score Fusion." *IEEE T-PAMI*, 2008, 30(2), 342-347. DOI: 10.1109/TPAMI.2007.70744
4. Misra, I.; Shrivastava, A.; Gupta, A.; Hebert, M. "Cross-stitch Networks for Multi-task Learning." *CVPR*, 2016. DOI: 10.1109/CVPR.2016.433
5. Neverova, N.; Wolf, C.; Taylor, G.; Nebout, F. "ModDrop: Adaptive Multi-modal Gesture Recognition." *IEEE T-PAMI*, 38(8):1692-1706, 2016. DOI: 10.1109/TPAMI.2015.2461544
6. Kendall, A.; Gal, Y. "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS*, 2017. [Foundational for aleatoric/epistemic separation]

### Early Fusion and Feature-Level Fusion

7. Soleymani, S., Torfi, A., Dawson, J., & Nasrabadi, N.M. "Generalized Bilinear Deep Convolutional Neural Networks for Multimodal Biometric Identification." *IEEE ICIP*, 2018. DOI: 10.1109/ICIP.2018.8451013
8. Soleymani, S., Dabouei, A., Kazemi, H., Dawson, J., & Nasrabadi, N.M. "Multi-Level Feature Abstraction from Convolutional Neural Networks for Multimodal Biometric Identification." *IJCB*, 2019.
9. Alay, N. & Al-Baity, H.H. "Deep Learning Approach for Multimodal Biometric Recognition System Based on Fusion of Iris, Face, and Finger Vein Traits." *Sensors*, 2020, 20(19), 5523. DOI: 10.3390/s20195523
10. Talreja, V., Soleymani, S., Valenti, M.C., & Nasrabadi, N.M. "Learning to Authenticate with Deep Multibiometric Hashing and Neural Network Decoding." *arXiv:1902.04149*, 2019.
11. Goh, Z.H. et al. "A framework for multimodal biometric authentication systems with template protection." *IEEE Access*, vol. 10, pp. 96388-96402, 2022. DOI: 10.1109/ACCESS.2022.3205413
12. Safavipour et al. "Deep Hybrid Multimodal Biometric Recognition System Based on Features-Level Deep Fusion of Five Biometric Traits." *Comp. Intelligence and Neuroscience*, 2023. DOI: 10.1155/2023/6443786

### Late Fusion and Score-Level Fusion

13. Nandakumar, K.; Chen, Y.; Jain, A.K.; Dass, S.C. "Quality-based Score Level Fusion in Multibiometric Systems." *ICPR*, 2006. DOI: 10.1109/ICPR.2006.1039
14. Medjahed, C. et al. "A Deep Learning-Based Multimodal Biometric System Using Score Fusion." *IAES Int. J. Artif. Intell.*, 2022, 11(1), 65-80. DOI: 10.11591/ijai.v11.i1.pp65-80
15. Shinde, K. & Kayte, C. "Multimodal Deep Learning Based Score Level Fusion Using Face and Fingerprint." *ACVAIT*, 2023. DOI: 10.2991/978-94-6463-196-8_13
16. Cherrat, E.; Alaoui, R.; Bouzahir, H. "Score Fusion of Finger Vein and Face for Human Recognition Based on CNN." *Int. J. Computers*, 2020, 19(1). DOI: 10.47839/ijc.19.1.1688

### Hybrid Fusion

17. Ren, H., Sun, L., Guo, J., & Han, C. "A Dataset and Benchmark for Multimodal Biometric Recognition Based on Fingerprint and Finger Vein." *IEEE T-IFS*, vol. 17, pp. 2030-2043, 2022. DOI: 10.1109/TIFS.2022.3175599
18. Al-Askar, A. et al. "Deep Learning-Based Fingerprint-Vein Biometric Fusion." *Applied Sciences*, vol. 15, no. 15, 8502, 2025. DOI: 10.3390/app15158502
19. Li, Zhang et al. "Joint Discriminative Feature Learning for Multimodal Finger Recognition." *Pattern Recognition*, 2021. DOI: 10.1016/j.patcog.2020.107704

### Transformers for Biometrics

20. Zhong, Y. & Deng, W. "Face Transformer for Recognition." *arXiv:2103.14803*, 2021.
21. Dan, K. et al. "TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective." *ICCV*, 2023. DOI: 10.1109/ICCV.2023
22. You, S. et al. "LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition." *arXiv:2501.13420*, 2025.
23. Chettaoui, F. et al. "ViT-GCT: Enhancing Vision Transformers with a Global Context Token for Face Recognition." *ICLR*, 2026. [OpenReview]
24. Grosz, S.A. & Jain, A.K. "AFR-Net: Attention-Driven Fingerprint Recognition Network." *IEEE T-BIOM*, 2024. DOI: 10.1109/TBIOM.2023.3345764
25. Qiu, Y. et al. "IFViT: Interpretable Fixed-Length Representation for Fingerprint Matching via Vision Transformer." *IEEE T-IFS*, 2024. DOI: 10.1109/TIFS.2024.3412270
26. Gazi et al. "Minutiae-Free Fingerprint Recognition via Vision Transformers: An Explainable Approach." *Applied Sciences*, 2026, 16(2), 1009. DOI: 10.3390/app16021009
27. Engelsma, J.J. et al. "Learning a Fixed-Length Fingerprint Representation (DeepPrint)." *IEEE T-PAMI*, 2021. DOI: 10.1109/TPAMI.2020.2987941
28. Grosz, S.A. et al. "Minutiae-Guided Vision Transformer for Fingerprint Recognition." *arXiv:2210.13994*, 2022.

### Cross-Modal Attention

29. Tiong, L.C.O. et al. "Flexible Biometrics Recognition: Bridging the Multimodality Gap Through Attention, Alignment and Prompt Tuning." *CVPR*, 2024, pp. 267-276. DOI: 10.1109/CVPR52733.2024.00033
30. Soleymani, S. et al. "Quality-Aware Multimodal Biometric Recognition." *IEEE TBIOM*, vol. 4, no. 1, pp. 97-116, 2021. DOI: 10.1109/TBIOM.2021.3131664
31. Yang, R., Zhang, Q., & Meng, L. "AuthFormer: Adaptive Multimodal Biometric Authentication Transformer for Middle-Aged and Elderly People." *arXiv:2411.05395*, 2024.
32. Gnanapraveen, S. et al. "Cross-Modal Transformers for Audio-Visual Person Recognition." *Odyssey 2024*, 2024.
33. Nagrani, A. et al. "Attention Bottlenecks for Multimodal Fusion." *NeurIPS*, 2021. arXiv:2107.00135

### Face Robustness

34. Kim, M., Jain, A.K., Liu, X. "AdaFace: Quality Adaptive Margin for Face Recognition." *CVPR*, 2022. arXiv:2204.00964
35. Terhorst, P. et al. "SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness." *CVPR*, 2020. DOI: 10.1109/CVPR42600.2020.00569
36. Boutros, F. et al. "CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability." *CVPR*, 2023. arXiv:2112.06592
37. Meng, Q. et al. "MagFace: A Universal Representation for Face Recognition and Quality Assessment." *CVPR*, 2021.
38. Terhorst, P. "QMagFace: Simple and Accurate Quality-Aware Face Recognition." *WACV*, 2023, pp. 3484-3494.
39. Low, C.Y. et al. "SlackedFace: Learning a Slacked Margin for Low-Resolution Face Recognition." *BMVC*, 2023.

### Fingerprint Robustness

40. He, Z. et al. "PFVNet: A Partial Fingerprint Verification Network Learned from Large Fingerprint Matching." *IEEE T-IFS*, 2022. DOI: 10.1109/TIFS.2022.3209869
41. Guan, X. et al. "Joint Identity Verification and Pose Alignment for Partial Fingerprints." *IEEE T-IFS*, 2025. DOI: 10.1109/TIFS.2024.3499505
42. Grosz, S.A. & Jain, A.K. "Latent Fingerprint Recognition: Fusion of Local and Global Embeddings (LFR-Net)." *IEEE T-IFS*, 2023. DOI: 10.1109/TIFS.2023.3314207
43. Grosz, S.A. et al. "C2CL: Contact to Contactless Fingerprint Matching." *IEEE T-IFS*, 2022. DOI: 10.1109/TIFS.2021.3134867
44. Priesnitz, J. et al. "Deep Learning-Based Fingerprint Quality Assessment." *BIOSIG*, 2025. [CITATION A VERIFIER]
45. Tabassi, E. et al. "NIST Fingerprint Image Quality (NFIQ 2)." *NISTIR 8382*, 2021.

### Missing Modality Handling

46. Ma, M. et al. "Are Multimodal Transformers Robust to Missing Modality?" *CVPR*, 2022, pp. 18177-18186. DOI: 10.1109/CVPR52688.2022.01764
47. Ma, M. et al. "SMIL: Multimodal Learning with Severely Missing Modality." *AAAI*, vol. 35, pp. 2302-2310, 2021. DOI: 10.1609/aaai.v35i3.16330
48. Lee, Y.-L. et al. "Multimodal Prompting With Missing Modalities for Visual Recognition." *CVPR*, 2023, pp. 14943-14952.
49. Reza, M.K. et al. "Robust Multimodal Learning via Cross-Modal Proxy Tokens." *arXiv:2501.17823*, 2025.
50. Reza, M.K. et al. "Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation." *arXiv:2310.03986*, 2023.
51. Zhang, J. et al. "Delivering Arbitrary-Modal Semantic Segmentation." *CVPR*, 2023.

### Uncertainty Quantification

52. Shi, W. & Jain, A.K. "Probabilistic Face Embeddings." *ICCV*, 2019.
53. Chang, J. et al. "Data Uncertainty Learning in Face Recognition." *CVPR*, 2020.
54. Brack, A. et al. "Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval." *NeurIPS*, 2023.
55. Han, Z. et al. "Trusted Multi-View Classification with Dynamic Evidential Fusion." *IEEE TPAMI*, 2022. DOI: 10.1109/TPAMI.2022.3210985
56. Conti, F. et al. "Assessing Uncertainty in Similarity Scoring: Performance & Fairness." *ICLR*, 2024.
57. He, S. et al. "A Survey on Uncertainty Quantification Methods for Deep Learning." *ACM Computing Surveys*, 2025. DOI: 10.1145/3786319

### Explainability

58. Lu, Y., Xu, Z., & Ebrahimi, T. "Towards Visual Saliency Explanations of Face Verification." *WACV*, 2024, pp. 4726-4735.
59. Huber, M. et al. "Efficient Explainable Face Verification Based on Similarity Score Argument Backpropagation." *WACV*, 2024, pp. 4736-4745.
60. Selvarani, S. & Rani, M.M.S. "Deep Learning-Powered Secure Multimodal Biometric Feature Fusion with Explainability and Real-Time Deployment." *ICAISDA*, 2025/2026. DOI: 10.2991/978-94-6239-616-6_71
61. Tan, H. & Kumar, A. "Minutiae Attention Network with Reciprocal Distance Loss for Contactless to Contact-based Fingerprint Identification." *IEEE T-IFS*, vol. 16, pp. 3299-3311, 2021. DOI: 10.1109/TIFS.2021.3076307
62. Lin, Y.S. et al. "xCos: An Explainable Cosine Metric for Face Verification Task." *ACM TOMM*, 2021. arXiv:2003.05383
63. Neto, P.C. et al. "Explainable Biometrics in the Age of Deep Learning." *arXiv:2208.09500*, 2022.
64. Rocha, R. et al. "Intrinsically-Interpretable Siamese Networks for Identity Recognition." *ICCV 2025 Workshop on CV4BIOM*.

### General Multimodal Fusion and Quality

65. Liang, Y. et al. "Deep Multimodal Data Fusion." *ACM Computing Surveys*, 2024. DOI: 10.1145/3649447
66. Zhu, X. et al. "A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition." *ICCV*, 2025. arXiv:2508.00053
67. Kolf, J.N. et al. "GraFIQs: Face Image Quality Assessment Using Gradient Magnitudes." *CVPRW*, 2024. DOI: 10.1109/CVPRW63382.2024.00156
68. Priesnitz, J. et al. "MCLFIQ: Mobile Contactless Fingerprint Image Quality." *IET Biometrics / arXiv*, 2023. arXiv:2304.14123
69. Ruder, S. et al. "Latent Multi-Task Architecture Learning (Sluice Networks)." *AAAI*, 2019.

### Surveys

70. Wu, R. et al. "Deep Multimodal Learning with Missing Modality: A Survey." *TMLR*, 2026.
71. He, S. et al. "A Survey on Uncertainty Quantification Methods for Deep Learning." *ACM Computing Surveys*, 2025. DOI: 10.1145/3786319
72. Neto, P.C. et al. "Explainable Biometrics in the Age of Deep Learning." *arXiv:2208.09500*, 2022.

---

*Document compiled from cross-verification of 10 dimension research files covering 150+ papers.*
*Confidence levels: High (>=2 independent dimensions confirm), Medium (1 authoritative source), Low (weak sourcing or preliminary).*
*Conflict zones represent genuine disagreements in the literature that require careful citation in academic writing.*
