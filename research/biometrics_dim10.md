## Dimension 10: Explainability (XAI) in Multimodal Biometric Systems

---

### Key Findings

- **Grad-CAM and its variants (Grad-CAM++, Score-CAM, SDD-CAM)** have been widely adopted for explaining face recognition decisions, but they produce less stable and meaningful explanation maps compared to newer perturbation-based methods like CorrRISE [^316^][^333^]. The SDD-CAM (Scaled Directed Divergence CAM) variant specifically targets fine localization of facial features for biometric authentication [^333^].

- **CorrRISE (Correlation-based Randomized Input Sampling for Explanation)**, proposed by Lu et al. at WACV 2024, represents the current state-of-the-art for explainable face verification, outperforming Grad-CAM, Grad-CAM++, LIME, RISE, MinPlus, and xFace in quantitative benchmarks [^317^][^366^]. CorrRISE generates saliency maps revealing both similar and dissimilar regions between face pairs and can explain failure cases (e.g., false accepts due to similar identities, head pose variations) [^316^].

- **SHAP and LIME have been applied to multimodal biometric fusion** in the work of Selvarani & Rani (ICAISDA 2025 / Atlantis Press 2026), providing the first thorough global-local interpretability analysis in biometric authentication [^56^][^397^]. Their system quantified modality contributions as: Face 48%, Fingerprint 31%, Palmprint 21% using SHAP global attribution, with LIME achieving 91.7% average local fidelity [^56^].

- **Minutiae attention mechanisms** for fingerprint recognition were pioneered by Tan & Kumar (IEEE T-IFS, 2021), who introduced a minutiae attention network that generates minutiae likelihood maps serving as attention maps to guide the recognition network toward distorted areas [^315^][^398^]. This represents one of the earliest explicit attempts to make fingerprint recognition explainable via attention visualization.

- **Vision Transformer attention maps for fingerprint recognition** demonstrate automatic alignment with classical minutiae regions (IoU = 0.41+-0.07) despite never being trained with minutiae supervision, as shown in a 2026 MDPI Applied Sciences study using DINOv2 self-supervised ViT [^48^]. This provides a new paradigm for explainable fingerprint recognition without explicit minutiae extraction.

- **Intrinsically interpretable Siamese networks** (ProtoSiamese and MaskSiamese) were introduced at ICCV 2025 Workshop on Computer Vision for Biometrics, offering fully interpretable facial verification by performing explicit patch-to-patch comparisons between face images, moving beyond post-hoc saliency maps [^336^].

- **xCos (explainable cosine metric)** by Lin et al. (2020) was one of the first learnable modules for explainable face verification, computing local similarities between face images with attention-weighted spatial explanations [^334^].

- **xSSAB (similarity score argument backpropagation)** by Huber et al. (WACV 2024) provides a training-free white-box approach that backpropagates the cosine similarity score through a Siamese FR setup to efficiently indicate similar and dissimilar areas [^389^][^390^].

- **The comprehensive survey "Explainable Biometrics in the Age of Deep Learning"** by Neto et al. (2022, arXiv:2208.09500) provides a systematic taxonomy of XAI techniques across biometric modalities, establishing a "xAI Ladder" with three rungs: (1) Gradient-based visualization, (2) Trainable/Regularized Post-hoc, (3) Ante-hoc interpretable design [^344^][^339^].

- **Cross-modal attention fusion** with explainability has been demonstrated by Selvarani & Rani, who use a multi-head cross-modal attention fusion network that dynamically allocates importance weights to fingerprint, palmprint, and face features, with the attention mechanism providing transparency into which modality matters most at any moment [^56^].

- **Explainable Presentation Attack Detection (PAD)** has received attention, with an attention-guided framework producing human-readable explanations via saliency maps and natural language justifications for fingerprint PAD decisions [^377^].

- **Graph Attention Networks (GAT) for multimodal biometric fusion** treat modalities as nodes in a fully connected graph, with learned inter-modal attention weights that dynamically redistribute confidence across modalities according to mutual consistency, providing inherent explainability through attention weights [^393^].

---

### Major Papers & Sources

| # | Author(s), Title, Venue, Year, DOI/URL |
|---|----------------------------------------|
| 1 | Lu, Y., Xu, Z., & Ebrahimi, T., "Towards Visual Saliency Explanations of Face Verification," IEEE/CVF WACV, 2024, pp. 4726-4735. https://openaccess.thecvf.com/content/WACV2024/html/Lu_Towards_Visual_Saliency_Explanations_of_Face_Verification_WACV_2024_paper.html |
| 2 | Lu, Y., Xu, Z., & Ebrahimi, T., "Towards A Comprehensive Visual Saliency Explanation Framework for AI-based Face Recognition Systems," arXiv:2407.05983, 2024. https://arxiv.org/abs/2407.05983 |
| 3 | Huber, M., Luu, A.T., Terhorst, P., & Damer, N., "Efficient Explainable Face Verification Based on Similarity Score Argument Backpropagation," IEEE/CVF WACV, 2024, pp. 4736-4745. https://openaccess.thecvf.com/content/WACV2024/html/Huber_Efficient_Explainable_Face_Verification_Based_on_Similarity_Score_Argument_Backpropagation_WACV_2024_paper.html |
| 4 | Shadman, R., Hou, D., Hussain, F., & Murshed, M.G.S., "Explainable Face Recognition via Improved Localization," Electronics, 2025, 14(14), 2745. https://doi.org/10.3390/electronics14142745 |
| 5 | Selvarani, S., & Rani, M.M.S., "Deep Learning-Powered Secure Multimodal Biometric Feature Fusion with Explainability and Real-Time Deployment," ICAISDA 2025 Proceedings, Atlantis Press, 2026. https://doi.org/10.2991/978-94-6239-616-6_71 |
| 6 | Tan, H., & Kumar, A., "Minutiae Attention Network with Reciprocal Distance Loss for Contactless to Contact-based Fingerprint Identification," IEEE T-IFS, vol. 16, pp. 3299-3311, 2021. https://doi.org/10.1109/TIFS.2021.3076307 |
| 7 | Lin, Y.S., Liu, Z.Y., Chen, Y.A., Wang, Y.S., Chang, Y.L., & Hsu, W.H., "xCos: An Explainable Cosine Metric for Face Verification Task," ACM TOMM, 17(3s), 2021. https://arxiv.org/abs/2003.05383 |
| 8 | Neto, P.C., Goncalves, T., Pinto, J.R., Silva, W., Sequeira, A.F., Ross, A., & Cardoso, J.S., "Explainable Biometrics in the Age of Deep Learning," arXiv:2208.09500, 2022. https://arxiv.org/pdf/2208.09500 |
| 9 | Rocha, R. et al., "Intrinsically-Interpretable Siamese Networks for Identity Recognition," ICCV 2025 Workshop on CV4BIOM, 2025. https://openaccess.thecvf.com/content/ICCV2025W/CV4BIOM/papers/Rocha_Intrinsically-Interpretable_Siamese_Networks_for_Identity_Recognition_ICCVW_2025_paper.pdf |
| 10 | "Minutiae-Free Fingerprint Recognition via Vision Transformers: An Explainable Approach," Applied Sciences, 2026, 16(2), 1009. https://doi.org/10.3390/app16021009 |
| 11 | Xu, Z., Lu, Y., & Ebrahimi, T., "Discriminative Deep Feature Visualization for Explainable Face Recognition," IEEE MMSP, 2023. https://infoscience.epfl.ch/bitstreams/677e4fd0-8786-42b1-998c-8d97b86a0ce2/download |
| 12 | Mery, D., "True Black-Box Explanation in Facial Analysis," CVPR 2022 Workshop on Biometrics, 2022. https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Mery_True_Black-Box_Explanation_in_Facial_Analysis_CVPRW_2022_paper.pdf |
| 13 | "An Attention-Guided Framework for Explainable Biometric Presentation Attack Detection," PMC/MDPI, 2022. https://pmc.ncbi.nlm.nih.gov/articles/PMC9102540/ |
| 14 | Harini, M., Beeram, D., & Shravan, "Multimodal Biometric Authentication Using Graph Attention Networks with Cross-Modal Triplet Learning," 2025-2026. |
| 15 | Ramesh, S., & Krishnaveni, V., "Adaptive Multimodal Biometric Recognition Framework Integrating Iris and Fingerprint Modalities for Robust and Interpretable Authentication," Lattice Science Publications, 2025. |
| 16 | Joshi, I., Kothari, R., et al., "Explainable Fingerprint ROI Segmentation Using Monte Carlo Dropout," IEEE/CVF WACV, 2021, pp. 60-69. [CITATION TO VERIFY] |

---

### Trends & Signals

- **Shift from post-hoc to ante-hoc explainability**: The field is moving from post-hoc saliency maps (Grad-CAM, LIME) toward intrinsically interpretable architectures (ProtoSiamese, MaskSiamese, xCos) that provide explanations as part of the model design rather than as afterthoughts [^336^][^334^].

- **Perturbation-based methods outperforming gradient-based methods**: CorrRISE consistently outperforms Grad-CAM and Grad-CAM++ in face verification explainability benchmarks (Deletion: 23.29% vs 50.65% for Grad-CAM on LFW; Insertion: 85.70% vs 65.00%) [^317^]. This trend suggests gradient-based methods may be less suitable for similarity-based tasks like face verification.

- **SHAP/LIME integration into multimodal biometric pipelines**: The combination of SHAP (global interpretability showing modality contributions) and LIME (local interpretability highlighting pixel-level influences) is emerging as a gold standard for multimodal biometric explainability, though currently only demonstrated in conference/workshop papers [^56^][^397^].

- **Attention-minutiae alignment**: Self-supervised Vision Transformers for fingerprint recognition spontaneously learn to attend to minutiae-rich regions (IoU up to 0.41), suggesting that minutiae-free approaches can still provide forensically meaningful explanations [^48^].

- **Cross-modal attention as explanation**: Transformer-inspired cross-modal attention fusion networks are being used not only to improve fusion performance but also to provide dynamic, input-dependent explanations of which modality contributed more to each decision [^56^].

- **Explainability for failure analysis**: There is growing interest in using explanation methods to understand failure cases -- false accepts due to very similar identities, false rejects due to head pose variations, occlusions -- to improve system design [^316^][^340^].

- **Benchmark datasets for explainable face recognition**: The introduction of Patch-LFW (by Huber et al. at WACV 2024) and WebFace-Occ (occluded face dataset) provides standardized evaluation protocols for quantitatively comparing explanation methods [^389^].

---

### Controversies & Conflicting Claims

- **Gradient-based vs. perturbation-based superiority**: Grad-CAM and Grad-CAM++ are the most widely cited XAI methods but consistently underperform perturbation-based approaches (CorrRISE, MinPlus, xFace) in face verification explainability benchmarks [^317^][^316^]. However, gradient-based methods (xSSAB, FGGB) are more computationally efficient, creating a trade-off between explanation quality and speed.

- **Attention maps as reliable explanations**: There is ongoing debate about whether attention maps provide faithful explanations. While ViT attention maps align with minutiae regions (IoU = 0.41) [^48^], some argue that attention weights do not always correlate with feature importance, and alternative methods like CheferCAM ( transformer-specific interpretability) may be needed for robust medical/biometric decision-making [^321^].

- **Post-hoc vs. ante-hoc explainability trade-offs**: Post-hoc methods (applied after training) are more flexible but may be less trustworthy; ante-hoc methods (built into the architecture) are more reliable but require specialized training. ProtoSiamese achieves slightly worse accuracy than black-box Siamese networks, raising questions about the accuracy-explainability trade-off [^336^].

- **Dissimilarity map limitations**: Perturbation-based methods (CorrRISE, xFace) struggle with generating accurate dissimilarity maps for non-matching pairs because the naturally low similarity scores provide limited signal for perturbation analysis. Lu et al. proposed a regularization term to address this, but it remains an open challenge [^316^].

- **SHAP modality contribution quantification**: Selvarani & Rani report face as the dominant modality (48% SHAP contribution), followed by fingerprint (31%) and palmprint (21%) [^56^]. However, this was tested on a specific dataset (LUTBIOS) with a specific architecture (cross-modal attention + CAE), and generalizability to other datasets and architectures is unknown.

---

### Research Gaps Identified

1. **BIMODAL explanations (face regions + minutiae points simultaneously)**: No existing work provides unified visual explanations that simultaneously highlight which facial regions AND which fingerprint minutiae points contributed to a multimodal biometric decision. Current multimodal explainability (e.g., Selvarani & Rani) quantifies modality-level contributions via SHAP but does not generate fused visual saliency maps overlaying both face and fingerprint explanations [^56^]. **This is the most significant gap.**

2. **Cross-modal attention visualization for face+fingerprint**: While cross-modal attention has been applied to face+fingerprint+palmprint fusion [^56^], there is no dedicated study on cross-modal attention visualization specifically for bimodal (face+fingerprint) systems that would reveal how the two modalities interact at the feature level.

3. **Explainability for multimodal biometric failure cases**: While Lu et al.'s CorrRISE framework explains failure cases in unimodal face recognition [^316^], there is no comparable work explaining why multimodal biometric systems fail -- e.g., when both face and fingerprint modalities jointly produce a false accept or false reject.

4. **Real-time explainability in multimodal biometric systems**: Most explainability methods (SHAP, LIME, CorrRISE) are computationally expensive and not suitable for real-time deployment. There is a need for efficient explainability techniques that can operate at biometric system inference speeds [^56^].

5. **Standardized benchmarks for multimodal biometric explainability**: While Patch-LFW exists for face verification explainability [^389^], there are no standardized benchmarks or evaluation protocols specifically for multimodal biometric explainability, making it impossible to compare different approaches fairly.

6. **Grad-CAM++ for fingerprint minutiae importance**: While Grad-CAM has been applied to face recognition and general fingerprint classification, there is no dedicated work applying Grad-CAM++ specifically to highlight which minutiae points (ridge endings, bifurcations) are most important for fingerprint matching decisions in deep learning models.

7. **Explainable deep learning for latent fingerprint matching**: Latent fingerprint recognition remains a challenging domain where explainability could provide significant value, but attention-based approaches (e.g., LatentPrintFormer) have not yet incorporated explicit explainability analysis [^216^].

8. **Intrinsically interpretable multimodal fusion architectures**: Current intrinsically interpretable approaches (ProtoSiamese, xCos) only address unimodal face recognition. No work has extended these concepts to multimodal biometric fusion.

9. **Temporal/sequential explainability for biometric systems**: Most explainability work focuses on static image-based decisions. There is limited work on explaining sequential or temporal biometric decisions (e.g., video-based face recognition + fingerprint swipe).

10. **Regulatory-compliant explainability for biometric AI**: The EU AI Act and similar regulations require explainability for high-risk AI systems including biometrics. There is a gap between research prototypes and deployable, auditable explainability frameworks that meet regulatory requirements [^340^].

---

### Detailed Evidence Log

#### Evidence 1: CorrRISE for Explainable Face Verification
**Claim**: CorrRISE outperforms Grad-CAM, Grad-CAM++, LIME, RISE, MinPlus, and xFace in face verification explainability benchmarks, particularly for similarity maps, and can explain both matching and non-matching decisions including failure cases.
**Source**: Lu, Y., Xu, Z., & Ebrahimi, T., "Towards Visual Saliency Explanations of Face Verification," IEEE/CVF WACV, 2024
**URL**: https://openaccess.thecvf.com/content/WACV2024/html/Lu_Towards_Visual_Saliency_Explanations_of_Face_Verification_WACV_2024_paper.html
**Date**: 2024
**Excerpt**: "The results show that CorrRISE consistently yields stable saliency maps, precisely highlighting the most similar regions between any given image pair. In comparison, the adapted Grad-CAM and Grad-CAM++ methods produce less stable and meaningful explanation maps. LIME and RISE tend to allocate a broad range of high-saliency pixels, making the importance map less precise."
**Confidence**: High

#### Evidence 2: CorrRISE Extended Framework for Face Identification
**Claim**: CorrRISE was extended to cover both face verification and face identification scenarios, providing the first comprehensive framework for explainable face recognition with quantitative evaluation methodology.
**Source**: Lu, Y., Xu, Z., & Ebrahimi, T., "Towards A Comprehensive Visual Saliency Explanation Framework for AI-based Face Recognition Systems," arXiv:2407.05983, 2024
**URL**: https://arxiv.org/abs/2407.05983
**Date**: 2024
**Excerpt**: "This manuscript contributes a comprehensive explanation framework for AI-based face recognition systems... a Correlation-based Randomized Input Sampling for Explanation (CorrRISE) algorithm is proposed... capable of providing saliency maps that adhere to the explainable face recognition definition and highlight similar and dissimilar regions between any input face images."
**Confidence**: High

#### Evidence 3: xSSAB - Efficient Gradient-Based Face Verification Explanation
**Claim**: xSSAB provides a training-free, efficient white-box approach to explain face verification by backpropagating similarity score arguments, and introduces Patch-LFW, the first explainable face verification benchmark.
**Source**: Huber, M., Luu, A.T., Terhorst, P., & Damer, N., "Efficient Explainable Face Verification Based on Similarity Score Argument Backpropagation," IEEE/CVF WACV, 2024
**URL**: https://openaccess.thecvf.com/content/WACV2024/html/Huber_Efficient_Explainable_Face_Verification_Based_on_Similarity_Score_Argument_Backpropagation_WACV_2024_paper.html
**Date**: 2024
**Excerpt**: "We propose a similarity score argument backpropagation (xSSAB) approach that supports or opposes the face-matching decision to visualize spatial maps that indicate similar and dissimilar areas as interpreted by the underlying FR model."
**Confidence**: High

#### Evidence 4: SDD-CAM for Explainable Face Recognition
**Claim**: SDD-CAM (Scaled Directed Divergence Class Activation Mapping) provides narrower and more precise localization of facial features compared to traditional CAM, improving transparency in AI-based face recognition decisions.
**Source**: Shadman, R., Hou, D., Hussain, F., & Murshed, M.G.S., "Explainable Face Recognition via Improved Localization," Electronics (MDPI), 2025
**URL**: https://doi.org/10.3390/electronics14142745
**Date**: 2025
**Excerpt**: "Our experiments show that the SDD Class Activation Map (CAM) highlights the relevant face features very specifically and accurately compared to the traditional CAM. The provided visual explanations with narrow localization of relevant features can ensure much-needed transparency and trust."
**Confidence**: High

#### Evidence 5: SHAP and LIME for Multimodal Biometric Fusion
**Claim**: SHAP and LIME were integrated into a multimodal biometric authentication pipeline (face + fingerprint + palmprint) for the first time, providing dual-layer explainability with quantified modality contributions.
**Source**: Selvarani, S., & Rani, M.M.S., "Deep Learning-Powered Secure Multimodal Biometric Feature Fusion with Explainability and Real-Time Deployment," ICAISDA 2025, Atlantis Press, 2026
**URL**: https://doi.org/10.2991/978-94-6239-616-6_71
**Date**: 2025/2026
**Excerpt**: "SHAP is used to give a big-picture view, showing how much each modality face, fingerprint, and palmprint contributes to the final decision... A SHAP-based analysis found that the face data had the biggest impact on verification results (48%), followed by fingerprints (31%) and palmprints (21%)."
**Confidence**: Medium (conference paper, limited dataset)

#### Evidence 6: Minutiae Attention Network for Fingerprint Recognition
**Claim**: A minutiae attention network with reciprocal distance loss enables contactless-to-contact-based fingerprint identification by generating minutiae likelihood maps that serve as attention maps, guiding the network to focus on distorted areas.
**Source**: Tan, H., & Kumar, A., "Minutiae Attention Network with Reciprocal Distance Loss for Contactless to Contact-based Fingerprint Identification," IEEE T-IFS, 2021
**URL**: https://doi.org/10.1109/TIFS.2021.3076307
**Date**: 2021
**Excerpt**: "Attention mechanism is introduced to guide the minutiae attention branch to concentrate on distorted areas and recover minutiae/features correspondence for contactless and contact-based fingerprint images from the same fingers."
**Confidence**: High

#### Evidence 7: xCos - Explainable Cosine Metric for Face Verification
**Claim**: xCos provides a learnable module that can be plugged into most face verification models to provide spatial explanations showing which parts of two input faces are similar, where the model pays attention, and how local similarities are weighted.
**Source**: Lin, Y.S. et al., "xCos: An Explainable Cosine Metric for Face Verification Task," ACM TOMM, 2021 (originally arXiv 2020)
**URL**: https://arxiv.org/abs/2003.05383
**Date**: 2020/2021
**Excerpt**: "With the help of xCos, we can see which parts of the two input faces are similar, where the model pays its attention to, and how the local similarities are weighted to form the output xCos score."
**Confidence**: High

#### Evidence 8: Explainable Biometrics Survey
**Claim**: A comprehensive survey establishes a taxonomy of XAI techniques for biometrics across modalities (face, fingerprint, iris) with a three-rung "xAI Ladder": gradient-based visualization, trainable/regularized post-hoc, and ante-hoc interpretable design.
**Source**: Neto, P.C. et al., "Explainable Biometrics in the Age of Deep Learning," arXiv:2208.09500, 2022
**URL**: https://arxiv.org/pdf/2208.09500
**Date**: 2022
**Excerpt**: "Biometric systems are everywhere, from smartphones and laptops to border control and other sensitive areas. Both experts and non-experts have constant contact with these systems, but do they really understand what is under the hood?"
**Confidence**: High

#### Evidence 9: Intrinsically Interpretable Siamese Networks
**Claim**: ProtoSiamese and MaskSiamese offer fully interpretable facial verification through explicit patch-to-patch comparisons, with MaskSiamese outperforming black-box Siamese networks while providing meaningful explanations.
**Source**: Rocha, R. et al., "Intrinsically-Interpretable Siamese Networks for Identity Recognition," ICCV 2025 Workshop on CV4BIOM
**URL**: https://openaccess.thecvf.com/content/ICCV2025W/CV4BIOM/papers/Rocha_Intrinsically-Interpretable_Siamese_Networks_for_Identity_Recognition_ICCVW_2025_paper.pdf
**Date**: 2025
**Excerpt**: "We propose intrinsically-interpretable strategies to match and compute the similarity between semantically-similar image regions... MaskSiamese outperforms a black-box Siamese network, while providing meaningful explanations."
**Confidence**: High

#### Evidence 10: Minutiae-Free ViT Fingerprint Recognition with Attention-Minutiae Alignment
**Claim**: A self-supervised DINOv2-based Vision Transformer for fingerprint recognition achieves spontaneous attention alignment with classical minutiae regions (IoU = 0.41+-0.07) despite never receiving minutiae supervision during training.
**Source**: "Minutiae-Free Fingerprint Recognition via Vision Transformers: An Explainable Approach," Applied Sciences (MDPI), 2026, 16(2), 1009
**URL**: https://doi.org/10.3390/app16021009
**Date**: 2026
**Excerpt**: "Across all samples, the Vision Transformer consistently concentrates attention on ridge flow transitions, high curvature regions, and bifurcation-like local patterns, despite having never been trained with any explicit minutiae supervision."
**Confidence**: High

#### Evidence 11: Discriminative Deep Feature Visualization
**Claim**: A face reconstruction-based explanation module reveals the correspondence between deep features and facial regions, and a novel saliency map generation algorithm provides explanations by highlighting similarity and dissimilarity regions.
**Source**: Xu, Z., Lu, Y., & Ebrahimi, T., "Discriminative Deep Feature Visualization for Explainable Face Recognition," IEEE MMSP, 2023
**URL**: https://infoscience.epfl.ch/bitstreams/677e4fd0-8786-42b1-998c-8d97b86a0ce2/download
**Date**: 2023
**Excerpt**: "This paper contributes to the problem of explainable face recognition by first conceiving a face reconstruction-based explanation module, which reveals the correspondence between the deep feature and the facial regions."
**Confidence**: High

#### Evidence 12: MinPlus - Black-Box Explanation for Facial Analysis
**Claim**: MinPlus is a model-agnostic saliency map methodology that can explain any facial analysis algorithm (verification, attribute recognition, detection) without manipulating the recognition model, achieving stable and interpretable results.
**Source**: Mery, D., "True Black-Box Explanation in Facial Analysis," CVPR 2022 Workshop on Biometrics
**URL**: https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Mery_True_Black-Box_Explanation_in_Facial_Analysis_CVPRW_2022_paper.pdf
**Date**: 2022
**Excerpt**: "MinPlus achieves saliency maps that are stable and interpretable to humans. In addition, our method shows promising results in comparison with other state-of-the-art methods like AVG, LIME and RISE."
**Confidence**: High

#### Evidence 13: Explainable Presentation Attack Detection
**Claim**: An attention-guided deep neural network framework for biometric PAD produces human-readable explanations (saliency maps and natural language) to accompany each decision, justifying algorithmic decisions and building trust with users.
**Source**: "An Attention-Guided Framework for Explainable Biometric Presentation Attack Detection," MDPI Sensors / PMC, 2022
**URL**: https://pmc.ncbi.nlm.nih.gov/articles/PMC9102540/
**Date**: 2022
**Excerpt**: "The justifications for decisions can be tracked and understood and help build trust with users, especially when unexpected rejection or acceptance decisions are produced."
**Confidence**: High

#### Evidence 14: Graph Attention Networks for Multimodal Biometric Fusion
**Claim**: Graph Attention Networks treat biometric modalities as nodes in a fully connected graph, computing learned inter-modal attention weights that provide inherent explainability through attention weight visualization.
**Source**: Harini, M., Beeram, D., & Shravan, "Multimodal Biometric Authentication Using Graph Attention Networks with Cross-Modal Triplet Learning," 2025-2026
**URL**: [Not publicly available]
**Date**: 2025/2026
**Excerpt**: "Treating the three biometric channels as nodes in a fully connected graph allows the fusion layer to dynamically redistribute confidence across modalities according to their mutual consistency."
**Confidence**: Medium (student thesis, small dataset of 75 subjects)

#### Evidence 15: Integrated Gradients for Multimodal Iris-Fingerprint Authentication
**Claim**: Integrated Gradients (IG) are used to guarantee integrity and openness in a multimodal biometric architecture integrating iris (MLP-Mixer) and fingerprint (VAE) modalities with cross-attention transformer fusion.
**Source**: Ramesh, S., & Krishnaveni, V., "Adaptive Multimodal Biometric Recognition Framework Integrating Iris and Fingerprint Modalities for Robust and Interpretable Authentication," Lattice Science Publications, 2025
**URL**: https://www.journals.latticescipub.com/index.php/ijsp/article/view/1121
**Date**: 2025
**Excerpt**: "By highlighting each feature's contribution to the final decision, Integrated Gradient (IG) are used to guarantee integrity and openness."
**Confidence**: Medium

---

### Summary: BIMODAL Explanations (Face + Fingerprint Simultaneously)

**Finding**: After extensive literature search across 10+ independent queries, **no published work was found that provides unified visual explanations simultaneously highlighting which facial regions AND which fingerprint minutiae points contributed to a bimodal (face+fingerprint) biometric decision.** This represents a significant and critical research gap.

The closest works are:
1. **Selvarani & Rani (2025/2026)** [^56^]: Quantifies modality-level contributions via SHAP (Face 48%, Fingerprint 31%, Palmprint 21%) but does NOT generate fused visual saliency maps. Uses cross-modal attention weights for dynamic modality importance but visual explainability is per-modality via LIME.
2. **Tan & Kumar (2021)** [^315^]: Provides minutiae attention maps for fingerprint-only recognition.
3. **Lu et al. (2024)** [^316^][^317^]: Provides CorrRISE saliency maps for face-only verification/identification.
4. **Harini et al. (2025/2026)** [^393^]: Uses GAT attention weights for interpretable fusion but does not generate unified visual explanations across modalities.

**A truly bimodal explanation system** that simultaneously overlays face saliency maps with fingerprint minutiae attention in a single unified visualization remains an open and high-impact research direction.
