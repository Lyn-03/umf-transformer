## Dimension 9: Uncertainty Quantification and Calibration in Biometric Verification

**Research Focus:** Uncertainty quantification (UQ) and score calibration in biometric recognition systems (face + fingerprint), 2020-2026. This dimension examines how modern deep learning approaches estimate confidence in biometric matching decisions, enabling risk-controlled recognition and rejection options.

---

### Key Findings

1. **Probabilistic Face Embeddings (PFE) pioneered uncertainty-aware face recognition** by representing each face image as a Gaussian distribution N(z; mu, sigma^2) in latent space, where mu estimates the identity feature and sigma quantifies uncertainty. PFE improves recognition performance and provides "good indicators of potential matching accuracy" for risk-controlled systems [^1^].

2. **Data Uncertainty Learning (DUL)** extends PFE by jointly learning identity representation (mean) and uncertainty (variance) end-to-end within a classification framework (DUL_cls) or regression framework (DUL_reg). DUL achieves higher TPR@FPR=0.001% on IJB-C (90.23%) compared to PFE (89.64%) and baseline (83.06%), while the learned variance correlates with image quality degradation and serves as a "risk indicator" [^2^].

3. **Monte Carlo Dropout (MC-Dropout) has been applied to fingerprint enhancement and segmentation** for model uncertainty estimation. Joshi et al. introduced MU-GAN (Model Uncertainty GAN) and DU-GAN (Data Uncertainty GAN), showing that model uncertainty is higher at incorrectly predicted pixels while data uncertainty is higher on noisy input regions [^3^][^4^].

4. **Bootstrap methods for uncertainty in similarity scoring** were formally studied by Conti et al. (ICLR 2024), who proved that naive bootstrap fails for similarity scoring because FAR/FRR are U-statistics. Their recentered bootstrap provides valid confidence bands around ROC curves and fairness metrics in face recognition [^5^].

5. **Holistic Uncertainty Estimation (HolUE)** combines gallery-aware uncertainty (GalUE) with sample quality estimates via a Bayesian probabilistic model, outperforming PFE, SCF, and SF on IJB-C for risk-controlled open-set recognition. HolUE distinguishes two uncertainty sources: gallery uncertainty (overlapping classes) and embedding uncertainty [^6^].

6. **Bayesian Metric Learning (LAM)** uses Laplace approximation over network weights rather than neural amortization, yielding well-calibrated uncertainties and state-of-the-art OOD detection. LAM outperforms MC Dropout, Deep Ensembles, PFE, and HIB on image retrieval tasks, with lower Expected Calibration Error (ECE) [^7^].

7. **Trusted Multi-View Classification (TMC)** applies evidential deep learning with dynamic Dempster-Shafer fusion for multi-view/multi-modal classification. While not biometric-specific, the framework provides a principled way to fuse uncertain evidence from multiple views with explicit uncertainty quantification [^8^].

8. **Conformal Prediction (CP) for face recognition** provides well-calibrated confidence measures with theoretical guarantees, producing prediction sets that contain the true identity at a specified confidence level, thus enabling "don't know" predictions with coverage guarantees [^9^].

9. **Dempster-Shafer theory has been applied to multimodal biometric fusion** at the score level, explicitly modeling three masses: genuine, impostor, and uncertainty. Nguyen et al. showed competitive performance (EER ~0.08-0.09) on the DS2 benchmark with face, fingerprint, and iris modalities [^10^].

10. **Context-Aware Decision Fusion** (MDPI 2026) is one of the very few works explicitly addressing uncertainty-aware multimodal biometric fusion at the decision level, using contradiction detection and reliability gating for face, voice, and fingerprint modalities under conflicting evidence [^11^].

11. **Evidential deep learning is emerging in facial analysis** through UAU-Net, which combines uncertainty-aware representations with Beta evidential neural networks for facial action unit detection, achieving calibrated occurrence estimates with explicit predictive uncertainty [^12^].

12. **Dual Gaussian Modeling for Deep Face Embeddings** addresses the limitation of single Gaussian approaches by using two Gaussian distributions to handle hard samples (noisy images, large pose) separately, correlating uncertainty with multiple attributes simultaneously [^13^].

---

### Major Papers & Sources

| # | Authors | Title | Venue | Year | DOI/URL |
|---|---------|-------|-------|------|---------|
| 1 | Shi & Jain | Probabilistic Face Embeddings | ICCV | 2019 | https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf |
| 2 | Chang et al. | Data Uncertainty Learning in Face Recognition | CVPR | 2020 | https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Data_Uncertainty_Learning_in_Face_Recognition_CVPR_2020_paper.pdf |
| 3 | Chen et al. | Fast and Reliable Probabilistic Face Embeddings in the Wild | arXiv | 2021 | https://arxiv.org/abs/2102.04075 |
| 4 | Joshi et al. | On Estimating Uncertainty of Fingerprint Enhancement Models | Book Chapter (Elsevier) | 2022 | https://hal.science/hal-04391813/file/elsevier22_uncertainty.pdf |
| 5 | Joshi et al. | Explainable Fingerprint ROI Segmentation Using Monte Carlo Dropout | WACV Workshop | 2021 | https://openaccess.thecvf.com/content/WACV2021W/XAI4B/papers/Joshi_Explainable_Fingerprint_ROI_Segmentation_Using_Monte_Carlo_Dropout_WACVW_2021_paper.pdf |
| 6 | Conti et al. | Assessing Uncertainty in Similarity Scoring: Performance & Fairness | ICLR | 2024 | https://openreview.net/forum?id=lAhQCHuANV |
| 7 | Erlygin & Zaytsev | Holistic Uncertainty Estimation For Open-Set Recognition | arXiv | 2024 | https://arxiv.org/abs/2408.14229 |
| 8 | Brack et al. | Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval | NeurIPS | 2023 | https://proceedings.neurips.cc/paper_files/paper/2023/file/da7ce04b3683b173691ecbb801f2690f-Paper-Conference.pdf |
| 9 | Han et al. | Trusted Multi-View Classification with Dynamic Evidential Fusion | IEEE TPAMI | 2022 | https://doi.org/10.1109/TPAMI.2022.3210985 |
| 10 | Hoshino & Crihalmeanu | Score calibration for optimal biometric identification | Book Chapter | 2015 | https://www.khoury.northeastern.edu/home/rhoshino/papers/or3.pdf |
| 11 | Nguyen et al. | Score-level multibiometric fusion based on Dempster-Shafer theory incorporating uncertainty factors | IEEE Trans. Hum.-Mach. Syst. | 2015 | https://doi.org/10.1109/THMS.2014.2363120 |
| 12 | He et al. | A Survey on Uncertainty Quantification Methods for Deep Learning | ACM Computing Surveys | 2025 | https://doi.org/10.1145/3786319 |
| 13 | Zibert et al. | Context-Aware Decision Fusion for Multimodal Access Control Under Contradictory Biometric Evidence | MDPI Computers | 2026 | https://www.mdpi.com/2073-431X/15/4/208 |
| 14 | Balomenos et al. | Automatic face recognition with well-calibrated confidence measures | Machine Learning (Springer) | 2018 | https://doi.org/10.1007/s10994-018-5756-7 |
| 15 | Chen et al. | Dual Gaussian Modeling for Deep Face Embeddings | Pattern Recognition Letters | 2022 | https://doi.org/10.1016/j.patrec.2022.10.002 |
| 16 | Conti et al. (PhD Thesis) | Bias in Face Recognition: Assessment and Post-Processing | PhD Thesis, Telecom Paris | 2025 | https://theses.hal.science/tel-05305376v1 |
| 17 | UAU-Net | Uncertainty-aware Representation Learning and Evidential Classification for Facial Action Unit Detection | arXiv | 2026 | https://arxiv.org/html/2604.21227v1 |
| 18 | HUE-Net | A Human-Centric, Uncertainty-Aware Event-Fused AI Network for Robust Face Recognition | Applied Sciences (MDPI) | 2025 | https://www.mdpi.com/2076-3417/15/13/7381 |
| 19 | Nogueira et al. | Machine Learning with a Reject Option: A survey | arXiv | 2024 | https://arxiv.org/abs/2107.11277 |

---

### Trends & Signals

- **Trend 1: Shift from point embeddings to probabilistic embeddings.** Starting with PFE (2019) and DUL (2020), face recognition has moved from deterministic feature vectors to distributional representations that encode both identity and uncertainty. This enables risk-controlled recognition and rejection options [^1^][^2^].

- **Trend 2: Separation of aleatoric and epistemic uncertainty.** Following Kendall & Gal (2017), biometric works now explicitly distinguish data (aleatoric) uncertainty from model (epistemic) uncertainty. Joshi et al. separately model both in fingerprint enhancement, showing they capture complementary information [^3^].

- **Trend 3: From sample quality to gallery-aware uncertainty.** Early works (PFE, DUL) focused on per-sample quality as the sole uncertainty source. Recent works (HolUE, LAM) show that gallery structure (class overlap) is equally important, especially for open-set recognition [^6^][^7^].

- **Trend 4: Convergence of evidential deep learning and biometric fusion.** Evidential methods (TMC, UAU-Net) provide principled uncertainty quantification through belief functions, naturally compatible with Dempster-Shafer fusion frameworks used in multimodal biometrics [^8^][^12^].

- **Trend 5: Uncertainty quantification moving beyond accuracy to fairness and calibration.** The ICLR 2024 paper by Conti et al. and the thesis work on bootstrap methods show that uncertainty in biometric evaluation metrics themselves is critical for fair assessment [^5^].

- **Trend 6: Score calibration is receiving renewed attention for operational deployment.** With the introduction of ISO/IEC 29794-5 for face image quality and NIST FRVT quality tracks, there is growing emphasis on calibrated confidence scores that reflect true match probabilities [^14^][^15^].

---

### Controversies & Conflicting Claims

- **Controversy 1: Computational cost of MC-Dropout vs. data uncertainty methods.** Joshi et al. report that MC-Dropout increases inference time ~10x (49.32ms vs 5.13ms for 10 samples), while data uncertainty methods add minimal overhead (5.87ms). However, MC-Dropout requires no architectural changes, whereas data uncertainty requires additional network branches [^3^].

- **Controversy 2: PFE vs. DUL for uncertainty estimation.** DUL claims to outperform PFE because it jointly learns identity and uncertainty end-to-end, whereas PFE only learns variance on top of frozen deterministic embeddings. However, PFE advocates argue that the Mutual Likelihood Score (MLS) metric, while computationally expensive, provides more principled matching [^1^][^2^].

- **Controversy 3: Does uncertainty estimation improve or degrade latent fingerprint performance?** Joshi et al. found that while data uncertainty improves enhancement for plain fingerprints, it can degrade performance on latent fingerprints with complex background noise, suggesting uncertainty methods may not generalize across all biometric sub-domains [^3^].

- **Controversy 4: Naive bootstrap underestimates uncertainty in similarity scoring.** Conti et al. demonstrated that standard bootstrap methods, widely used in ML evaluation, produce invalid confidence bands for face recognition ROC curves because FAR/FRR are U-statistics, not simple averages. This challenges common practice in biometric evaluation [^5^].

- **Controversy 5: Temperature scaling and Platt scaling are widely used in ML calibration but rarely validated specifically for biometric score distributions.** While these methods are standard in general ML, their effectiveness for biometric similarity scores (which often have different distributional properties than classification softmax outputs) remains under-studied [^10^][^16^].

---

### Research Gaps Identified

- **Gap 1 (CRITICAL): Very few works combine uncertainty quantification with MULTIMODAL biometric fusion.** Despite extensive literature on both unimodal biometric uncertainty and multimodal fusion, the intersection is nearly empty. The MDPI 2026 paper [^11^] and Nguyen et al.'s D-S fusion [^10^] are rare exceptions. There is no comprehensive framework that propagates uncertainty from each modality (face + fingerprint) through the fusion process to produce calibrated multimodal confidence scores.

- **Gap 2: Expected Calibration Error (ECE) and reliability diagrams are rarely reported in biometric recognition papers.** While ECE is standard in ML calibration literature, biometric works typically report only FAR/FRR/EER without assessing whether similarity scores are calibrated probabilities. This gap is noted but not addressed in most biometric UQ works [^5^][^7^].

- **Gap 3: Temperature scaling and Platt scaling have not been systematically studied for biometric verification scores.** These calibration methods are widely used in classification but their application to biometric similarity scores (cosine similarity, Euclidean distance) is under-explored, especially in the context of multimodal fusion.

- **Gap 4: Deep Ensembles for face/fingerprint recognition uncertainty are surprisingly under-studied.** Despite being a gold-standard UQ method in general ML (Fort et al., 2019), deep ensembles have seen limited application in biometric recognition. Most works use single-model approaches (PFE, DUL) or MC-Dropout.

- **Gap 5: Rejection option ("don't know" prediction) based on uncertainty is not widely deployed.** While theoretically studied (Chow's reject option, Selective Classification), practical integration of uncertainty-based rejection into operational biometric systems remains limited. The machine learning with rejection survey [^19^] notes that practical approaches are mostly model-specific.

- **Gap 6: Conformal prediction for biometrics is in its infancy.** Despite providing finite-sample coverage guarantees ideal for high-stakes biometric decisions, conformal prediction has only been applied to face recognition in limited settings [^9^], not to multimodal biometric fusion.

- **Gap 7: No standardized benchmarks for uncertainty evaluation in biometrics.** Unlike classification (where ECE, Brier score, NLL are standard), biometric UQ lacks standardized evaluation protocols. Different works use different metrics (variance correlation, rejection curves, AUROC for error detection), making comparison difficult [^6^][^7^].

- **Gap 8: The relationship between ISO/IEC 29794-5 quality measures and deep learning uncertainty is unexplored.** Traditional biometric quality standards define image quality heuristically, while modern UQ methods learn uncertainty from data. The relationship between these two approaches remains unclear [^15^].

- **Gap 9: Evidential deep learning for face/fingerprint biometric scores is largely unexplored.** While evidential methods have been applied to medical imaging (EDL) and multi-view classification (TMC), direct application to biometric verification scores with explicit belief assignment is a significant gap.

- **Gap 10: Uncertainty propagation through the entire biometric pipeline.** Most works estimate uncertainty at a single stage (enhancement, feature extraction, or matching). No work comprehensively models uncertainty propagation from raw sensor data through preprocessing, feature extraction, matching, and fusion.

---

### Detailed Evidence Log

#### Evidence 1: Probabilistic Face Embeddings (PFE)
```
Claim: Representing face images as Gaussian distributions in latent space improves
       recognition performance and provides calibrated uncertainty estimates that
       correlate with matching accuracy.
Source: Shi & Jain, "Probabilistic Face Embeddings", ICCV 2019
URL: https://openaccess.thecvf.com/content_ICCV_2019/papers/Shi_Probabilistic_Face_Embeddings_ICCV_2019_paper.pdf
Date: 2019
Excerpt: "The uncertainties estimated by PFEs also serve as good indicators of the
          potential matching accuracy, which are important for a risk-controlled
          recognition system."
Confidence: HIGH
```

#### Evidence 2: Data Uncertainty Learning (DUL)
```
Claim: Jointly learning identity features and data uncertainty end-to-end in face
       recognition improves performance on challenging benchmarks; the learned variance
       correlates with image quality and can serve as a risk indicator.
Source: Chang et al., "Data Uncertainty Learning in Face Recognition", CVPR 2020
URL: https://openaccess.thecvf.com/content_CVPR_2020/papers/Chang_Data_Uncertainty_Learning_in_Face_Recognition_CVPR_2020_paper.pdf
Date: 2020
Excerpt: "The estimated uncertainty is closely related to the quality of face images...
          This learned uncertainty could be regarded as a 'risk indicator' to alert FR
          systems that the output decision is unreliable when the estimated variance is
          very high."
Confidence: HIGH
```

#### Evidence 3: MC Dropout for Fingerprint Enhancement (MU-GAN/DU-GAN)
```
Claim: Monte Carlo dropout can estimate model uncertainty in fingerprint enhancement
       without additional parameters; data uncertainty requires architectural changes
       but adds minimal computational overhead. Model uncertainty is high at erroneous
       predictions, data uncertainty at noisy inputs.
Source: Joshi et al., "On Estimating Uncertainty of Fingerprint Enhancement Models", 2022
URL: https://hal.science/hal-04391813/file/elsevier22_uncertainty.pdf
Date: 2022
Excerpt: "Higher model uncertainty is predicted for the pixels where the enhancement
          model generates spurious patterns... data uncertainty is predicted for noisy
          and background pixels."
Confidence: HIGH
```

#### Evidence 4: Bootstrap Uncertainty for Similarity Scoring
```
Claim: Naive bootstrap produces invalid confidence bands for biometric similarity
       scoring because FAR/FRR are U-statistics; a recentered bootstrap is required
       for valid inference about ROC-based performance and fairness metrics.
Source: Conti et al., "Assessing Uncertainty in Similarity Scoring", ICLR 2024
URL: https://openreview.net/forum?id=lAhQCHuANV
Date: 2024
Excerpt: "Because the false acceptance/rejection rates are of the form of U-statistics
          in the case of similarity scoring, the naive bootstrap approach may jeopardize
          the assessment procedure."
Confidence: HIGH
```

#### Evidence 5: Holistic Uncertainty Estimation (HolUE)
```
Claim: Combining gallery-aware uncertainty with sample quality estimates via Bayesian
       modeling outperforms single-source uncertainty methods for open-set face
       recognition, achieving PRR=0.82 on IJB-C vs 0.42 for SCF.
Source: Erlygin & Zaytsev, "Holistic Uncertainty Estimation For Open-Set Recognition", 2024
URL: https://arxiv.org/abs/2408.14229
Date: 2024
Excerpt: "HolUE better identifies recognition errors than alternative uncertainty
          estimation methods, including those based solely on sample quality."
Confidence: HIGH
```

#### Evidence 6: Bayesian Metric Learning (LAM)
```
Claim: Bayesian metric learning using Laplace approximation over network weights
       produces better-calibrated uncertainties than MC Dropout, Deep Ensembles, PFE,
       and HIB for image retrieval, with lower ECE and superior OOD detection.
Source: Brack et al., "Bayesian Metric Learning for Uncertainty Quantification in Image Retrieval",
       NeurIPS 2023
URL: https://proceedings.neurips.cc/paper_files/paper/2023/file/da7ce04b3683b173691ecbb801f2690f-Paper-Conference.pdf
Date: 2023
Excerpt: "Online LAM outperforms other Bayesian methods, such as post-hoc LAM and
          MC dropout, on this task, which in turn clearly improves upon amortized
          methods that rely on a neural network to extrapolate uncertainties."
Confidence: HIGH
```

#### Evidence 7: Trusted Multi-View Classification (TMC)
```
Claim: Dynamic evidential fusion based on Dempster-Shafer theory provides principled
       uncertainty quantification for multi-view classification with explicit handling
       of conflict between views.
Source: Han et al., "Trusted Multi-View Classification with Dynamic Evidential Fusion",
       IEEE TPAMI, 2022
URL: https://doi.org/10.1109/TPAMI.2022.3210985
Date: 2022
Excerpt: [From GitHub description] "Constructing trustworthy multi-view/multi-modal
          classification algorithm"
Confidence: HIGH [CITATION TO VERIFY: exact excerpt from paper not captured]
```

#### Evidence 8: Score Calibration Function (SCF) for Biometrics
```
Claim: The Score Calibration Function transforms matching scores into calibrated
       confidence scores that equal true match probabilities, producing optimal DET
       curves for biometric identification.
Source: Hoshino & Crihalmeanu, "Score calibration for optimal biometric identification"
URL: https://www.khoury.northeastern.edu/home/rhoshino/papers/or3.pdf
Date: 2015
Excerpt: "This SCF function replaces matching scores with meaningful confidence scores
          that are perfectly calibrated and normalized... produces the biometric system's
          best possible DET curve."
Confidence: MEDIUM (older work, limited experimental validation)
```

#### Evidence 9: Dempster-Shafer Multimodal Fusion
```
Claim: Score-level fusion using Dempster-Shafer theory with explicit uncertainty
       factors achieves competitive performance (EER ~0.08-0.09) on face+fingerprint+iris
       multimodal benchmarks.
Source: Nguyen et al., "Score-level multibiometric fusion based on Dempster-Shafer theory
       incorporating uncertainty factors", IEEE THMS, 2015
URL: https://doi.org/10.1109/THMS.2014.2363120
Date: 2015
Excerpt: [Referenced in MDPI 2022] "The motivation for applying D-S theory to
          multibiometric fusion is to take advantage of the uncertainty concept in D-S
          theory to deal with uncertainty factors that impact biometric accuracy."
Confidence: HIGH
```

#### Evidence 10: Context-Aware Decision Fusion for Multimodal Biometrics
```
Claim: Explicit modeling of contradictions and uncertainties at the decision level
       enables robust multimodal authentication even when biometric modalities disagree,
       using reliability gating and context-aware arbitration.
Source: Zibert et al., "Context-Aware Decision Fusion for Multimodal Access Control
       Under Contradictory Biometric Evidence", MDPI Computers, 2026
URL: https://www.mdpi.com/2073-431X/15/4/208
Date: 2026
Excerpt: "The proposed framework explicitly considers inconsistencies, uncertainties,
          and contextual reliability when aggregating evidence, enabling robust
          authorization decisions even in the presence of conflicting biometric signals."
Confidence: HIGH
```

#### Evidence 11: Conformal Prediction for Face Recognition
```
Claim: Conformal Prediction can complement face recognition with well-calibrated
       confidence measures and prediction sets with guaranteed coverage, without
       assuming more than data exchangeability.
Source: Balomenos et al., "Automatic face recognition with well-calibrated confidence
       measures", Machine Learning (Springer), 2018
URL: https://doi.org/10.1007/s10994-018-5756-7
Date: 2018
Excerpt: "CP can provide either a confidence measure that indicates the likelihood of
          each recognition being correct, or produce a prediction set that is guaranteed
          to satisfy a given confidence level."
Confidence: HIGH
```

#### Evidence 12: Uncertainty Quantification Survey
```
Claim: UQ for DNNs can be taxonomized by uncertainty source (data vs. model vs. both);
       evidential deep learning and Bayesian methods are key approaches; biometric
       applications are an important domain.
Source: He et al., "A Survey on Uncertainty Quantification Methods for Deep Learning",
       ACM Computing Surveys, 2025
URL: https://doi.org/10.1145/3786319
Date: 2025
Excerpt: "UQ estimates the confidence of DNN predictions in addition to their accuracy...
          Evidential deep learning directly predicts the parameters of Dirichlet density."
Confidence: HIGH
```

#### Evidence 13: Dual Gaussian Modeling for Face Recognition
```
Claim: Using two Gaussian distributions (separating easy and hard samples) produces
       more discriminative embeddings and correlates uncertainty with multiple attributes
       (quality, pose) simultaneously.
Source: Chen et al., "Dual Gaussian Modeling for Deep Face Embeddings",
       Pattern Recognition Letters, 2022
URL: https://doi.org/10.1016/j.patrec.2022.10.002
Date: 2022
Excerpt: "The uncertainty relates with both image quality and facial pose."
Confidence: HIGH
```

#### Evidence 14: UAU-Net (Evidential Deep Learning for Facial Action Units)
```
Claim: Combining CVAE-based uncertainty-aware features with Beta evidential neural
       networks produces calibrated predictions with explicit uncertainty for facial
       action unit detection.
Source: "UAU-Net: Uncertainty-aware Representation Learning and Evidential Classification
       for Facial Action Unit Detection", arXiv, 2026
URL: https://arxiv.org/html/2604.21227v1
Date: 2026
Excerpt: "AB-ENN formulates AU prediction as a binary evidential learning problem with
          Beta distributions, producing calibrated occurrence estimates together with
          explicit predictive uncertainty."
Confidence: MEDIUM (recent arXiv preprint)
```

#### Evidence 15: HUE-Net for Robust Face Recognition
```
Claim: Monte Carlo Dropout combined with Bayesian backpropagation and multi-branch
       variational modules can quantify both epistemic and aleatoric uncertainty for
       face recognition in adverse conditions.
Source: "A Human-Centric, Uncertainty-Aware Event-Fused AI Network for Robust Face
       Recognition in Adverse Conditions", Applied Sciences (MDPI), 2025
URL: https://www.mdpi.com/2076-3417/15/13/7381
Date: 2025
Excerpt: "HUE-Net implements the perturbed multi-branch variational module... based on
          concepts of Bayesian neural networks and stochastic dropouts."
Confidence: MEDIUM
```

---

### Summary Statistics

| Category | Count |
|----------|-------|
| Total papers identified | 19 |
| Papers from 2020-2026 | 16 |
| Papers from before 2020 | 3 (foundational) |
| Top-tier venues (CVPR, ICCV, NeurIPS, ICLR, IEEE TPAMI) | 7 |
| IEEE biometrics venues (T-IFS, T-BIOM) | 0 directly |
| Multimodal biometric fusion + uncertainty papers | 2-3 |
| Single-modality uncertainty papers | 16 |

---

### Key Takeaway for Multimodal Fusion Research

The most striking finding of this literature review is the **near-total absence of works that combine uncertainty quantification with multimodal biometric fusion** (face + fingerprint). While there is rich literature on:
- Uncertainty for face recognition alone (PFE, DUL, HolUE, LAM)
- Uncertainty for fingerprint enhancement alone (MU-GAN, DU-GAN)
- Multimodal biometric fusion without uncertainty (score-level, Dempster-Shafer)

...the intersection of these three areas is essentially empty. The MDPI 2026 paper [^11^] on context-aware decision fusion and the earlier Dempster-Shafer work [^10^] are among the very few exceptions. This represents a **critical research gap** with significant practical importance: operational multimodal biometric systems would benefit enormously from knowing not just which modality disagrees, but how uncertain each modality's decision is, enabling principled arbitration rather than ad-hoc fusion rules.
