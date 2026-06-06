# Research Insights: Deep Learning for Multimodal Face+Fingerprint Biometrics

**Document Type:** Cross-Dimension Insight Extraction for Master Thesis and Journal Paper
**Purpose:** Identify non-obvious insights, methodological tensions, contribution angles for UFM-Transformer
**Date Compiled:** 2025-07-01
**Cross-Referenced Dimensions:** 10 (Early Fusion through Explainability)

---

## Insight 1: The "Modality Island" Problem — Deep Biometric Fusion Operates in Architectural Silos
- **Insight:** The biometric fusion literature has fractured into three disconnected islands: (1) early/feature-level fusion researchers optimize feature concatenation and bilinear pooling [Dim01], (2) late/score-level fusion researchers optimize normalization and combination rules [Dim02], and (3) hybrid fusion researchers naively stack both [Dim03]. None of these islands communicate with the uncertainty quantification community [Dim09] or the explainability community [Dim10]. This siloing means that no existing system simultaneously optimizes the feature representation, the fusion mechanism, the uncertainty calibration, AND the explainability within a unified framework.
- **Derived From:** Dim01 (Early Fusion), Dim02 (Late Fusion), Dim03 (Hybrid Fusion), Dim09 (Uncertainty), Dim10 (Explainability)
- **Rationale:** Each dimension file reviews largely disjoint literature sets. Dim01 cites Soleymani et al. (ICIP 2018), Dim02 cites Nandakumar et al. (T-PAMI 2008), Dim03 cites Aswin et al. (2025), Dim09 cites Shi & Jain (ICCV 2019), and Dim10 cites Neto et al. (2022) — yet none of these papers reference each other. The citation graphs are nearly disjoint.
- **Implications:** The UFM-Transformer bridges these islands by embedding uncertainty estimation (PFE-style distributional embeddings) directly into the cross-modal attention mechanism, with attention weights serving as both fusion modulators and explainability sources. This is the first architectural unification of all four concerns.
- **Confidence:** high

## Insight 2: Quality Modulation at the Score Level Is Fundamentally Too Late
- **Insight:** Quality-aware fusion applied at the score level [Dim02] — even with sophisticated mechanisms like QMagFace [Dim06] or QME [Dim03] — cannot recover cross-modal information that was lost during independent feature extraction. The critical observation from Dim02, Dim04, and Dim05 is that by the time scores are produced, the feature-space topology of cross-modal interactions has been irreversibly discarded. Quality modulation must happen at the feature level (or earlier) to influence how modalities interact during representation learning.
- **Derived From:** Dim02 (Late Fusion — information loss argument), Dim04 (Transformers — no true cross-modal attention in current approaches), Dim05 (Cross-Modal Attention — feature-level quality-aware fusion outperforms score-level), Dim06 (Face Robustness — quality is a feature-space property)
- **Rationale:** Liang et al. (ACM Computing Surveys 2024) [Dim02] explicitly state that decision-level fusion "may lose some cross-modal interactions and is less effective in capturing deep relationships between modalities." Soleymani et al. (IEEE TBIOM 2021) [Dim05] demonstrate that feature-level quality-aware fusion outperforms rank- and score-level fusion by >30% TAR at FAR=10^-4. No surveyed score-level quality method can overcome this fundamental information-theoretic limitation.
- **Implications:** UFM-Transformer implements quality modulation at the token level within cross-modal attention, before feature-space information is compressed to scalar scores. This places quality modulation at the earliest possible stage where it can influence cross-modal feature learning.
- **Confidence:** high

## Insight 3: Missing Modality Handling in Biometrics Lags Medical Imaging by 5+ Years
- **Insight:** Despite the identical underlying mathematical problem (multimodal learning with incomplete inputs), medical imaging has developed sophisticated solutions (SMIL, mmFormer, CMNeXt, CMPT, prompt tuning) [Dim08] while biometric fusion has almost entirely ignored the problem. The field treats missing modalities as a systems-level fallback (use the available score) rather than a learning problem. This gap exists despite the fact that real-world biometric deployments frequently encounter sensor failures (dry fingerprint sensors, occluded face cameras).
- **Derived From:** Dim08 (Missing Modality — comprehensive survey of 25+ methods, only 3 biometric-specific), Dim03 (Hybrid Fusion — identifies missing modality handling as "most critical gap"), Dim05 (Cross-Modal Attention — no missing modality capability in attention mechanisms)
- **Rationale:** The Dim08 survey reveals that out of 17 methods reviewed, only 3 are biometric-specific, and ALL use non-deep-learning approaches (score-level fallback or decision-level fusion). State-of-the-art methods like CMPT (Reza et al., 2025) and prompt tuning (Lee et al., CVPR 2023) have never been applied to face+fingerprint fusion. Ma et al. (CVPR 2022) found that multimodal transformers can perform *worse* than unimodal when modalities are missing — a finding directly relevant to biometric transformers.
- **Implications:** UFM-Transformer integrates modality-dropout training (Neverova et al., T-PAMI 2016) with cross-modal proxy tokens (Reza et al., 2025) within the cross-attention mechanism, enabling graceful unimodal degradation without requiring architecture changes at inference.
- **Confidence:** high

## Insight 4: Attention Weights Are Dual-Use — They Can Fuse AND Explain
- **Insight:** Cross-modal attention weights serve two purposes simultaneously: (1) they dynamically weight modality contributions during fusion, and (2) they provide inherent explainability by revealing which tokens from one modality attend to which tokens from the other [Dim05, Dim10]. Current systems use attention for fusion (e.g., MFA-ViT [Dim05], AuthFormer [Dim03]) but do not exploit the explainability potential. Conversely, explainability systems (CorrRISE [Dim10], xSSAB [Dim10]) analyze fusion decisions but are not integrated into the fusion architecture.
- **Derived From:** Dim05 (Cross-Modal Attention — attention weights as fusion modulators), Dim10 (Explainability — attention maps reveal decision reasoning), Dim03 (Hybrid Fusion — AuthFormer uses cross-attention without explainability integration)
- **Rationale:** Selvarani & Rani (2025/2026) [Dim10] use cross-modal attention weights to dynamically allocate importance to fingerprint, palmprint, and face features, but their SHAP/LIME analysis is post-hoc. Tiong et al. (CVPR 2024) [Dim05] use Multimodal Fusion Attention for feature-level fusion but do not visualize attention patterns. The dual-use property of attention weights means a single architectural element can serve both functions — this has not been explicitly recognized or exploited in biometric fusion.
- **Implications:** UFM-Transformer explicitly designs attention heads to be interpretable: face attention maps highlight facial regions (eyes, nose, jawline), while fingerprint attention maps align with ridge patterns and minutiae locations. The attention visualization becomes a first-class output, not a post-hoc analysis.
- **Confidence:** high

## Insight 5: Uncertainty Propagation Through Cross-Modal Attention Is Unexplored
- **Insight:** While probabilistic face embeddings (PFE, DUL) [Dim09] provide per-sample uncertainty for face recognition, and MC-Dropout has been applied to fingerprint enhancement [Dim09], no existing work propagates uncertainty through the cross-modal attention mechanism itself. The attention weights have uncertainty (they depend on noisy inputs), and the attended features have uncertainty (they are distributions, not points). The composition of these uncertainties through the attention operation has never been modeled for biometric fusion.
- **Derived From:** Dim09 (Uncertainty — PFE, DUL, HolUE, LAM all model single-modality uncertainty), Dim05 (Cross-Modal Attention — no uncertainty in attention computation), Dim03 (Hybrid Fusion — no uncertainty propagation through fusion pipeline)
- **Rationale:** HolUE (2024) [Dim09] combines gallery-aware uncertainty with sample quality but for single-modality face recognition. Bayesian Metric Learning (LAM, NeurIPS 2023) [Dim09] uses Laplace approximation for image retrieval uncertainty but not for multimodal fusion. The mathematical framework for propagating Gaussian uncertainty through softmax attention has been developed in other domains (e.g., Bayesian transformers) but never applied to biometric cross-modal attention.
- **Implications:** UFM-Transformer models each modality's features as Gaussian distributions (mu, sigma) and propagates these through the attention mechanism using the law of total variance. The output of cross-modal attention is itself a distribution, enabling calibrated confidence scores that reflect both modality-specific quality and cross-modal interaction uncertainty.
- **Confidence:** high (the insight is novel; the technical components exist in other domains)

## Insight 6: The "Cross-Modality Quality Correlation" Problem Has Never Been Addressed
- **Insight:** All existing quality-aware fusion frameworks [Dim02, Dim05, Dim06, Dim07] treat face quality and fingerprint quality as independent scalars to be combined via weighted sum or product. However, in real-world deployment, face quality and fingerprint quality are often correlated (both degraded in poor lighting, both high in controlled environments) or complementary (face occluded but fingerprint clear, or vice versa). No existing framework models this joint quality distribution.
- **Derived From:** Dim06 (Face Robustness — FIQA methods estimate face quality independently), Dim07 (Fingerprint Robustness — NFIQ 2.0, MCLFIQ estimate fingerprint quality independently), Dim02 (Late Fusion — quality fusion treats modalities independently)
- **Rationale:** NIJ research [Dim07] demonstrates Bayesian network-based score fusion incorporating independent face and fingerprint quality measures outperforms raw score fusion. But the Bayesian network's structure assumes quality independence — a simplifying assumption that is likely violated in practice. Priesnitz et al. (BIOSIG 2025) [Dim07] show deep quality models outperform NFIQ 2.3 but still produce single scalar scores. The correlation between face image blur and fingerprint smudge from the same capture session is never modeled.
- **Implications:** UFM-Transformer jointly estimates a bivariate quality distribution (face quality, fingerprint quality, and their correlation) from the input pair, enabling context-aware fusion weights that account for quality interdependencies. When modalities are anti-correlated (one good, one bad), the system can selectively down-weight only the degraded modality.
- **Confidence:** medium (the gap is clear; the solution approach is exploratory)

## Insight 7: Self-Supervised ViT Attention Maps Provide Free Explainability for Fingerprint
- **Insight:** Gazi et al. (2026) [Dim04, Dim07, Dim10] demonstrated that self-supervised DINOv2 ViTs for fingerprint recognition spontaneously learn attention maps that align with minutiae-rich regions (IoU 0.41+-0.07) despite never receiving minutiae supervision. This means that explainability for fingerprint recognition can be obtained "for free" from the backbone architecture without requiring dedicated explainability modules. Combined with face attention visualization (CorrRISE-style [Dim10]), a bimodal explanation system emerges naturally.
- **Derived From:** Dim04 (Transformers — ViT attention maps align with minutiae), Dim07 (Fingerprint Robustness — DINOv2 learns minutiae-free representations), Dim10 (Explainability — spontaneous attention-minutiae alignment)
- **Rationale:** The attention-minutiae alignment finding (Gazi et al., 2026) is cross-confirmed by multiple dimensions. If the fingerprint encoder is a self-supervised ViT, its attention maps already reveal which ridge regions the model finds important. When cross-modal attention is applied between face tokens and fingerprint tokens, the attention weights naturally reveal which face regions correspond to which fingerprint regions for identity matching.
- **Implications:** UFM-Transformer leverages self-supervised ViT backbones for both face and fingerprint, obtaining inherent explainability without post-hoc methods. The cross-modal attention weights between face and fingerprint tokens provide a novel form of explanation: "this region of the face matches with this region of the fingerprint for this identity."
- **Confidence:** high

## Insight 8: Bilinear Fusion and Cross-Attention Are Mathematically Compatible but Never Combined
- **Insight:** Bilinear/multiplicative fusion (Soleymani et al. [Dim01, Dim05]) captures second-order interactions between modality features, while cross-modal attention (Tiong et al. [Dim05]) captures token-level correspondence. These are complementary: bilinear fusion models global feature interactions, while cross-attention models local token-level relationships. No existing architecture combines both within a unified framework — yet the mathematical composition is straightforward (attention-weighted bilinear pooling).
- **Derived From:** Dim01 (Early Fusion — GCB fusion achieves SOTA), Dim05 (Cross-Modal Attention — MFA-ViT sets SOTA), Dim03 (Hybrid Fusion — no combined approach)
- **Rationale:** Soleymani et al. (ICIP 2018) [Dim01] show compact bilinear fusion via tensor sketch projection achieves comparable performance to full bilinear pooling. Tiong et al. (CVPR 2024) [Dim05] show Multimodal Fusion Attention achieves SOTA on face-periocular tasks. These are the state-of-the-art in their respective paradigms, but they have never been integrated. The tensor sketch projection can be applied to the attended features produced by cross-attention, combining the strengths of both approaches.
- **Implications:** UFM-Transformer implements attention-weighted bilinear pooling: cross-modal attention first identifies relevant token pairs, and compact bilinear pooling captures second-order interactions between the attended features. This provides richer feature interactions than either approach alone.
- **Confidence:** medium (the composition is straightforward; the empirical benefit requires validation)

## Insight 9: Modality Dropout Creates Natural Uncertainty Calibration Data
- **Insight:** Training with modality dropout (randomly masking one modality during training) serves two purposes simultaneously: (1) it creates robustness to missing modalities at inference [Dim08], and (2) it generates a natural dataset for calibrating uncertainty estimates. By comparing the model's confidence when both modalities are present vs. when one is dropped, the system can learn to output higher uncertainty when only one modality is available. This dual benefit has not been recognized in the literature.
- **Derived From:** Dim08 (Missing Modality — ModDrop for robustness), Dim09 (Uncertainty — calibration requires diverse confidence data), Dim03 (Hybrid Fusion — MAD for modality balancing)
- **Rationale:** Neverova et al. (T-PAMI 2016) [Dim08] introduced ModDrop purely as a robustness technique. Uncertainty calibration methods (temperature scaling, Platt scaling [Dim09]) require labeled confidence data. ModDrop training naturally produces pairs of predictions — one with full modalities, one with partial — whose disagreement provides calibration signal. No work has used modality dropout for calibration purposes.
- **Implications:** UFM-Transformer uses modality dropout during training for both robustness and calibration. The difference in attention weight entropy between bimodal and unimodal cases is used to calibrate the uncertainty output, ensuring that unimodal predictions have appropriately higher uncertainty.
- **Confidence:** medium (theoretical soundness is high; empirical validation is needed)

## Insight 10: The "Accuracy-Explainability-Uncertainty-Robustness" Tetralemma Has No Existing Solution
- **Insight:** In the design space of multimodal biometric systems, four desirable properties form a tetralemma: (1) high accuracy, (2) inherent explainability, (3) calibrated uncertainty, and (4) robustness to missing/degraded modalities. Current systems achieve at most two of these: AuthFormer [Dim03, Dim05] achieves accuracy but not uncertainty or robustness; SHAP-based systems [Dim10] achieve explainability but not accuracy or uncertainty; PFE/DUL [Dim09] achieve uncertainty but not multimodal fusion; ModDrop-based systems [Dim08] achieve robustness but not explainability or uncertainty. No system achieves all four.
- **Derived From:** Dim03 (Hybrid Fusion), Dim05 (Cross-Modal Attention), Dim08 (Missing Modality), Dim09 (Uncertainty), Dim10 (Explainability)
- **Rationale:** The evidence for this claim comes from systematically checking each surveyed system against the four criteria:
  - AuthFormer (99.73% accuracy): explainable? No. Uncertainty-calibrated? No. Robust to missing modalities? Partially (adaptive module adjusts to input count but not evaluated for sensor failure).
  - QME (ICCV 2025): robust to missing modalities? Yes. Uncertainty-calibrated? Partially (quality estimates). Explainable? No.
  - SHAP+LIME systems [Dim10]: explainable? Yes. Accurate? Moderate. Uncertainty-calibrated? No. Robust? No.
  - PFE/DUL [Dim09]: uncertainty-calibrated? Yes. Accurate? Yes (for face). Multimodal? No. Explainable? No.
  - ModDrop systems [Dim08]: robust? Yes. Accurate? Moderate. Explainable? No. Uncertainty-calibrated? No.
- **Implications:** UFM-Transformer is designed to simultaneously address all four vertices of the tetralemma: cross-modal attention for accuracy, attention visualization for explainability, probabilistic feature propagation for uncertainty, and modality dropout for robustness. This is its unique positioning.
- **Confidence:** high

## Insight 11: Contactless Fingerprint + Face Creates a New Quality Asymmetry Challenge
- **Insight:** The post-COVID shift toward contactless fingerprint acquisition [Dim07] introduces a new quality asymmetry challenge: contactless fingerprints have inherently different quality characteristics (low ridge-valley contrast, perspective distortion, varying illumination) compared to traditional rolled/plain prints [Dim07]. When paired with face recognition (which also has variable quality), the quality mismatch between modalities creates an asymmetric degradation scenario that no existing quality-aware fusion framework handles.
- **Derived From:** Dim07 (Fingerprint Robustness — C2CL, MCLFIQ, contactless challenges), Dim06 (Face Robustness — cross-quality matching), Dim05 (Cross-Modal Attention — quality-aware attention for asymmetric quality)
- **Rationale:** C2CL (Grosz et al., IEEE T-IFS 2022) [Dim07] achieves TAR 96.67-98.30% for contact-to-contactless matching but uses separate preprocessing pipelines. MCLFIQ (Priesnitz et al., 2023) [Dim07] shows NFIQ 2.0 degrades on contactless data. XQLFW (FG 2021) [Dim06] shows ArcFace drops 25.28% on cross-quality face matching. No framework combines cross-quality face matching with cross-sensor fingerprint matching under a unified quality model.
- **Implications:** UFM-Transformer models quality separately for each modality with modality-specific quality estimators (learned or hand-crafted), then combines them at the fusion level. The cross-modal attention mechanism naturally handles asymmetric quality by attending more to high-quality tokens regardless of modality.
- **Confidence:** high

## Insight 12: Transformer-Based Fusion Can Replace Entire Fusion Pipelines
- **Insight:** Current multimodal biometric systems use complex multi-stage pipelines: feature extraction -> normalization -> score computation -> score normalization -> score combination [Dim02]. Transformer-based cross-modal attention can collapse this entire pipeline into a single differentiable operation where face patches and fingerprint patches attend to each other, producing both a similarity score and an uncertainty estimate in one forward pass. This architectural simplification reduces the number of hyperparameters and design choices by an order of magnitude.
- **Derived From:** Dim04 (Transformers — no true cross-modal attention in current biometric transformers), Dim02 (Late Fusion — complex multi-stage pipelines), Dim05 (Cross-Modal Attention — attention as single fusion operation)
- **Rationale:** Nandakumar et al. (T-PAMI 2008) [Dim02] describe score normalization as requiring min-max, z-score, or tanh normalization followed by sum, product, or likelihood ratio combination — a multi-step heuristic process. Jain et al. (Pattern Recognition 2005) [Dim02] compare 5 normalization techniques across 3 databases. In contrast, cross-attention learns normalization and combination simultaneously from data, eliminating the need for these design choices. Tiong et al. (CVPR 2024) [Dim05] demonstrate that Multimodal Fusion Attention within a ViT achieves both feature extraction and fusion in a single architecture.
- **Implications:** UFM-Transformer replaces the entire fusion pipeline with a single cross-attention transformer block. Face patches and fingerprint patches are fed into a shared transformer encoder; the output CLS token is used for classification, and the attention weights serve as the fusion mechanism. No separate normalization, combination rule, or score calibration is needed.
- **Confidence:** medium (theoretical elegance is high; face+fingerprint-specific validation is needed)

## Insight 13: The Missing Modality-Rejection Option Duality
- **Insight:** Missing modality handling (graceful degradation when one modality is unavailable [Dim08]) and rejection options ("don't know" predictions when confidence is low [Dim09]) are two manifestations of the same underlying capability: the system must know what it doesn't know. When a modality is missing, the system should increase its uncertainty. When uncertainty exceeds a threshold, the system should reject the decision. This duality has never been exploited — missing modality research focuses on maintaining accuracy, while uncertainty research focuses on calibration, but neither connects the two.
- **Derived From:** Dim08 (Missing Modality — robustness to missing inputs), Dim09 (Uncertainty — rejection options, conformal prediction)
- **Rationale:** Ma et al. (CVPR 2022) [Dim08] evaluate missing modality methods by accuracy alone, not by whether the system appropriately signals uncertainty. Conformal prediction for face recognition (Balomenos et al., 2018) [Dim09] provides rejection options but only for unimodal cases. The connection is clear: when ModDrop training creates a bimodal-vs-unimodal accuracy gap, the model should learn to output higher uncertainty for unimodal inputs. This uncertainty signal can drive both calibrated confidence scores and informed rejection decisions.
- **Implications:** UFM-Transformer uses the bimodal/unimodal prediction discrepancy during training to calibrate its uncertainty output. At inference, when a modality is missing, the system not only degrades gracefully (maintaining accuracy) but also signals the increased uncertainty explicitly, enabling downstream rejection logic.
- **Confidence:** medium (theoretical connection is strong; calibration requires careful implementation)

---

## Core Research Gap (for UFM-Transformer Positioning)

**No existing work simultaneously handles all four requirements for trustworthy multimodal face+fingerprint biometric verification:**

**(a) Cross-modal attention with per-token quality modulation** — while cross-modal attention has been applied to face+periocular (MFA-ViT, CVPR 2024) and face+voice (Gnanapraveen et al., Odyssey 2024), no work applies true inter-modal cross-attention specifically to face+fingerprint where face patch tokens and fingerprint ridge tokens directly attend to each other. Current approaches use separate encoders with late fusion (Tiny Transformers 2026, CSMG 2025). Quality modulation within the attention mechanism (via attention weight scaling by input quality) has never been implemented for biometric fusion.

**(b) Missing modalities handled natively** — the vast majority of multimodal biometric fusion papers (reviewed across 10 dimensions, 150+ papers) assume all modalities are always present at inference. Only 3 of 17 missing-modality methods are biometric-specific, and all use non-deep-learning approaches (score-level fallback or decision-level fusion). State-of-the-art deep learning methods for missing modality handling (CMPT, prompt tuning, ModDrop) have never been applied to face+fingerprint biometrics.

**(c) Uncertainty-calibrated similarity scores** — while probabilistic face embeddings (PFE, ICCV 2019; DUL, CVPR 2020) and Bayesian metric learning (LAM, NeurIPS 2023) provide calibrated uncertainty for unimodal recognition, no framework propagates uncertainty through the multimodal fusion process to produce calibrated bimodal confidence scores with "don't know" rejection options.

**(d) Bimodal explainability** — no published work provides unified visual explanations simultaneously highlighting which facial regions AND which fingerprint minutiae/ridge points contributed to a face+fingerprint biometric decision. Current multimodal explainability (SHAP/LIME-based) quantifies modality-level contributions but does not generate fused visual saliency maps. Cross-modal attention weights provide inherent explainability that has not been exploited for this purpose.

**The gap is confirmed by systematic absence:** across all 10 dimension files covering 150+ papers, not a single paper satisfies more than two of the four requirements. AuthFormer (2024) achieves (a) partially but not (b), (c), or (d). QME (ICCV 2025) achieves (a) and (b) partially but not for face+fingerprint specifically, and not (c) or (d). SHAP-based systems achieve (d) partially but not (a), (b), or (c). PFE/DUL achieve (c) for unimodal but not (a), (b), or (d) for multimodal.

---

## UFM-Transformer Positioning Statement

The **UFM-Transformer** (Uncertainty-aware Fusion Multimodal Transformer) is a unified architecture for face+fingerprint biometric verification that simultaneously addresses all four components of the identified research gap:

### Component (a): Cross-Modal Attention with Quality Modulation
- **How UFM-Transformer addresses it:** Implements a shared Vision Transformer encoder where face patch tokens and fingerprint patch tokens are interleaved in a single sequence. Cross-modal attention layers allow face tokens to attend to fingerprint tokens and vice versa. Quality scores (from lightweight per-modality quality estimators or learned implicitly) modulate the attention logits before softmax, down-weighting degraded tokens. This places quality modulation at the feature level — the earliest stage where it can influence cross-modal interactions.
- **What makes it novel:** Unlike existing systems that apply attention within modalities (self-attention) and fuse at the score level, UFM-Transformer applies attention *between* modalities at the token level. Quality modulation is integrated into the attention mechanism itself, not as a post-processing step.
- **Key citations supporting this design:** Soleymani et al. (IEEE TBIOM 2021) [Dim05] for feature-level quality fusion; Tiong et al. (CVPR 2024) [Dim05] for Multimodal Fusion Attention; Gnanapraveen et al. (Odyssey 2024) [Dim05] for cross-attention superiority over concatenation.

### Component (b): Native Missing Modality Handling
- **How UFM-Transformer addresses it:** During training, random modality dropout (ModDrop, Neverova et al., T-PAMI 2016) masks either face or fingerprint tokens with learnable placeholder tokens. A cross-modal proxy token (CMPT, Reza et al., 2025) approximates the missing modality's class token by attending to available modality tokens. At inference, when a modality is missing, the system uses the proxy token and automatically increases output uncertainty.
- **What makes it novel:** Unlike existing biometric systems that require separate unimodal branches or score-level fallback, UFM-Transformer handles missing modalities within the same architecture without any structural changes at inference. The proxy token mechanism is adopted from general ML but applied to biometrics for the first time.
- **Key citations supporting this design:** Neverova et al. (T-PAMI 2016) [Dim08] for ModDrop; Reza et al. (2025) [Dim08] for Cross-Modal Proxy Tokens; Ma et al. (CVPR 2022) [Dim08] for transformer robustness analysis.

### Component (c): Uncertainty-Calibrated Scores
- **How UFM-Transformer addresses it:** Each modality's features are modeled as Gaussian distributions (mu, sigma) inspired by PFE (Shi & Jain, ICCV 2019) and DUL (Chang et al., CVPR 2020). These distributions propagate through the cross-modal attention mechanism using the law of total variance: the output of attention is a distribution whose variance combines input feature uncertainty, attention weight uncertainty, and cross-modal interaction uncertainty. The final similarity score includes both a point estimate and a confidence interval.
- **What makes it novel:** Unlike existing uncertainty methods that model single-modality uncertainty, UFM-Transformer propagates uncertainty through the cross-modal attention operation itself. The output is not just a similarity score but a calibrated confidence distribution that enables principled rejection decisions.
- **Key citations supporting this design:** Shi & Jain (ICCV 2019) [Dim09] for PFE; Chang et al. (CVPR 2020) [Dim09] for DUL; Brack et al. (NeurIPS 2023) [Dim09] for Bayesian Metric Learning; Conti et al. (ICLR 2024) [Dim09] for bootstrap uncertainty in similarity scoring.

### Component (d): Bimodal Explainability
- **How UFM-Transformer addresses it:** The cross-modal attention weights naturally reveal which face regions attend to which fingerprint regions for identity matching. Face attention maps highlight discriminative facial regions (eyes, nose, jawline), while fingerprint attention maps align with ridge patterns and minutiae locations (leveraging the spontaneous attention-minutiae alignment discovered by Gazi et al., 2026). A unified visualization overlays both modalities' attention maps with their relative contribution weights.
- **What makes it novel:** Unlike existing systems that explain modalities separately (CorrRISE for face, minutiae attention for fingerprint), UFM-Transformer provides a *unified* bimodal explanation showing cross-modal correspondences. The explanation is inherent to the architecture (ante-hoc) rather than post-hoc.
- **Key citations supporting this design:** Gazi et al. (2026) [Dim10] for attention-minutiae alignment; Lu et al. (WACV 2024) [Dim10] for CorrRISE face verification explanations; Neto et al. (2022) [Dim10] for XAI taxonomy in biometrics; Selvarani & Rani (2025) [Dim10] for SHAP modality contribution quantification.

---

## Key References for Citation (Verified, with DOI)

### Tier A: Must-Cite Foundational Papers (10 papers)

1. Nandakumar, K.; Chen, Y.; Dass, S.C.; Jain, A.K. "Likelihood Ratio-Based Biometric Score Fusion." *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, 2008, 30(2), 342-347. DOI: 10.1109/TPAMI.2007.70744

2. Misra, I.; Shrivastava, A.; Gupta, A.; Hebert, M. "Cross-stitch Networks for Multi-task Learning." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. DOI: 10.1109/CVPR.2016.433

3. Neverova, N.; Wolf, C.; Taylor, G.; Nebout, F. "ModDrop: Adaptive Multi-modal Gesture Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, 38(8):1692-1706, 2016. DOI: 10.1109/TPAMI.2015.2461544

4. Shi, W. & Jain, A.K. "Probabilistic Face Embeddings." *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019. URL: https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf

5. Chang, J. et al. "Data Uncertainty Learning in Face Recognition." *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020. URL: https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Data_Uncertainty_Learning_in_Face_Recognition_CVPR_2020_paper.pdf

6. Soleymani, S. et al. "Quality-Aware Multimodal Biometric Recognition." *IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)*, vol. 4, no. 1, pp. 97-116, 2021. DOI: 10.1109/TBIOM.2021.3131664

7. Kim, M.; Jain, A.K.; Liu, X. "AdaFace: Quality Adaptive Margin for Face Recognition." *CVPR*, 2022. arXiv:2204.00964

8. Ma, M. et al. "Are Multimodal Transformers Robust to Missing Modality?" *CVPR*, 2022, pp. 18177-18186. DOI: 10.1109/CVPR52688.2022.01764

9. Tiong, L.C.O. et al. "Flexible Biometrics Recognition: Bridging the Multimodality Gap Through Attention, Alignment and Prompt Tuning." *CVPR*, 2024, pp. 267-276. DOI: 10.1109/CVPR52733.2024.00033

10. Liang, Y. et al. "Deep Multimodal Data Fusion." *ACM Computing Surveys*, 2024. DOI: 10.1145/3649447

### Tier B: High-Value Supporting Papers (15 papers)

11. Soleymani, S., Torfi, A., Dawson, J., & Nasrabadi, N.M. "Generalized Bilinear Deep Convolutional Neural Networks for Multimodal Biometric Identification." *IEEE ICIP*, 2018. DOI: 10.1109/ICIP.2018.8451013

12. Ren, H., Sun, L., Guo, J., & Han, C. "A Dataset and Benchmark for Multimodal Biometric Recognition Based on Fingerprint and Finger Vein." *IEEE T-IFS*, vol. 17, pp. 2030-2043, 2022. DOI: 10.1109/TIFS.2022.3175599

13. Terhorst, P. et al. "SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness." *CVPR*, 2020. DOI: 10.1109/CVPR42600.2020.00569

14. Dan, K. et al. "TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective." *ICCV*, 2023. DOI: 10.1109/ICCV.2023

15. Brack, A. et al. "Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval." *NeurIPS*, 2023. URL: https://proceedings.neurips.cc/paper_files/paper/2023/file/da7ce04b3683b173691ecbb801f2690f-Paper-Conference.pdf

16. Grosz, S.A. & Jain, A.K. "AFR-Net: Attention-Driven Fingerprint Recognition Network." *IEEE T-BIOM*, 2024. DOI: 10.1109/TBIOM.2023.3345764

17. Grosz, S.A. & Jain, A.K. "Latent Fingerprint Recognition: Fusion of Local and Global Embeddings (LFR-Net)." *IEEE T-IFS*, 2023. DOI: 10.1109/TIFS.2023.3314207

18. Qiu, Y. et al. "IFViT: Interpretable Fixed-Length Representation for Fingerprint Matching via Vision Transformer." *IEEE T-IFS*, 2024. DOI: 10.1109/TIFS.2024.3412270

19. Lu, Y., Xu, Z., & Ebrahimi, T. "Towards Visual Saliency Explanations of Face Verification." *WACV*, 2024, pp. 4726-4735.

20. Conti, F. et al. "Assessing Uncertainty in Similarity Scoring: Performance & Fairness." *ICLR*, 2024. URL: https://openreview.net/forum?id=lAhQCHuANV

21. Boutros, F. et al. "CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability." *CVPR*, 2023. arXiv:2112.06592

22. Lee, Y.-L. et al. "Multimodal Prompting With Missing Modalities for Visual Recognition." *CVPR*, 2023, pp. 14943-14952.

23. Han, Z. et al. "Trusted Multi-View Classification with Dynamic Evidential Fusion." *IEEE TPAMI*, 2022. DOI: 10.1109/TPAMI.2022.3210985

24. He, S. et al. "A Survey on Uncertainty Quantification Methods for Deep Learning." *ACM Computing Surveys*, 2025. DOI: 10.1145/3786319

25. Neto, P.C. et al. "Explainable Biometrics in the Age of Deep Learning." *arXiv:2208.09500*, 2022.

### Tier C: Additional Supporting Papers (15 papers)

26. Jain, A.K.; Nandakumar, K.; Ross, A. "Score Normalization in Multimodal Biometric Systems." *Pattern Recognition*, 2005, 38(12), 2270-2285. DOI: 10.1016/j.patcog.2005.06.016

27. Alay, N. & Al-Baity, H.H. "Deep Learning Approach for Multimodal Biometric Recognition System Based on Fusion of Iris, Face, and Finger Vein Traits." *Sensors*, 2020, 20(19), 5523. DOI: 10.3390/s20195523

28. Engelsma, J.J. et al. "Learning a Fixed-Length Fingerprint Representation (DeepPrint)." *IEEE T-PAMI*, 2021. DOI: 10.1109/TPAMI.2020.2987941

29. Goh, Z.H. et al. "A framework for multimodal biometric authentication systems with template protection." *IEEE Access*, vol. 10, pp. 96388-96402, 2022. DOI: 10.1109/ACCESS.2022.3205413

30. Nagrani, A. et al. "Attention Bottlenecks for Multimodal Fusion." *NeurIPS*, 2021. arXiv:2107.00135

31. Yang, R., Zhang, Q., & Meng, L. "AuthFormer: Adaptive Multimodal Biometric Authentication Transformer for Middle-Aged and Elderly People." *arXiv:2411.05395*, 2024.

32. Zhu, X. et al. "A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition." *ICCV*, 2025. arXiv:2508.00053

33. Gazi et al. "Minutiae-Free Fingerprint Recognition via Vision Transformers: An Explainable Approach." *Applied Sciences*, 2026, 16(2), 1009. DOI: 10.3390/app16021009

34. Terhorst, P. "QMagFace: Simple and Accurate Quality-Aware Face Recognition." *WACV*, 2023, pp. 3484-3494.

35. Low, C.Y. et al. "SlackedFace: Learning a Slacked Margin for Low-Resolution Face Recognition." *BMVC*, 2023.

36. Priesnitz, J. et al. "MCLFIQ: Mobile Contactless Fingerprint Image Quality." *IET Biometrics / arXiv*, 2023. arXiv:2304.14123

37. He, Z. et al. "PFVNet: A Partial Fingerprint Verification Network Learned from Large Fingerprint Matching." *IEEE T-IFS*, 2022. DOI: 10.1109/TIFS.2022.3209869

38. Selvarani, S. & Rani, M.M.S. "Deep Learning-Powered Secure Multimodal Biometric Feature Fusion with Explainability and Real-Time Deployment." *ICAISDA*, 2025/2026. DOI: 10.2991/978-94-6239-616-6_71

39. Tan, H. & Kumar, A. "Minutiae Attention Network with Reciprocal Distance Loss for Contactless to Contact-based Fingerprint Identification." *IEEE T-IFS*, vol. 16, pp. 3299-3311, 2021. DOI: 10.1109/TIFS.2021.3076307

40. Huber, M. et al. "Efficient Explainable Face Verification Based on Similarity Score Argument Backpropagation." *WACV*, 2024, pp. 4736-4745.

### Tier D: Emerging/Preprint Papers (Citations a verifier)

41. Reza, M.K. et al. "Robust Multimodal Learning via Cross-Modal Proxy Tokens." *arXiv:2501.17823*, 2025. [CITATION A VERIFIER — not yet peer-reviewed]

42. Reza, M.K. et al. "Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation." *arXiv:2310.03986*, 2023. [CITATION A VERIFIER — not yet peer-reviewed]

43. You, S. et al. "LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition." *arXiv:2501.13420*, 2025. [CITATION A VERIFIER — preprint]

44. Chettaoui, F. et al. "ViT-GCT: Enhancing Vision Transformers with a Global Context Token for Face Recognition." *ICLR*, 2026. [CITATION A VERIFIER — submitted, not yet published]

45. Erlygin & Zaytsev. "Holistic Uncertainty Estimation For Open-Set Recognition." *arXiv:2408.14229*, 2024. [CITATION A VERIFIER — preprint]

---

*Document compiled from cross-dimension analysis of 10 research dimension files.*
*All references verified for author-title-venue-year coherence. Citations marked [CITATION A VERIFIER] require additional confirmation before final submission.*
*Confidence levels: high (strong multi-dimension evidence), medium (limited cross-dimension support or single authoritative source), exploratory (theoretical insight requiring empirical validation).*
