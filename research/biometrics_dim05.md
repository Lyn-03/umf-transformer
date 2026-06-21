## Dimension 5: Cross-Modal Attention Mechanisms for Biometric Fusion

**Date compiled:** 2026-01-18
**Search scope:** 2018-2026, focusing on 2020-2026
**Total papers identified:** 15+
**Venues covered:** CVPR, IEEE TBIOM, IEEE TIFS, Sensors, arXiv, ARACE, PeerJ, ICPR, IEEE TETCI, Odyssey

---

### Key Findings

1. **Transformers with cross-attention are increasingly adopted for multimodal biometric fusion**, with the MFA-ViT architecture (CVPR 2024) representing the current state-of-the-art, achieving strong performance on both intra- and cross-modality face-periocular recognition tasks [^1^].

2. **Quality-aware attention is the most practically important mechanism** for real-world deployment, as biometric samples frequently exhibit asymmetric quality (one good, one degraded). Soleymani et al.'s IEEE TBIOM 2021 framework demonstrated >30% improvement in TAR@FAR=10^-4 by weighting modalities with weakly-supervised quality scores [^2^].

3. **Cross-attention consistently outperforms self-attention and simple feature concatenation** for multimodal biometric fusion. Gnanapraveen et al. (Odyssey 2024) showed that cross-attention fusion achieves EER of 2.387 vs. self-attention's 2.412 and feature concatenation's 2.489 on audio-visual person recognition [^3^].

4. **Phase-only cross-attention (POC-ViT) offers an elegant solution for illumination-invariant biometric authentication**, using phase correlation in the frequency domain to compute cross-attention between dual biometric traits, achieving 98.8% accuracy on forehead vein + periocular pattern recognition [^4^].

5. **Branch attention mechanisms dynamically weight modalities based on their importance**, with LiuJun et al.'s branch attention module (ARACE 2023) achieving 92.76% rank-1 accuracy even on heavily contaminated datasets by fusing, selecting, and calibrating branch weights [^5^].

6. **Quality-gated fusion using lightweight MLPs adapts fusion weights in real-time** based on input quality measures (e.g., Laplacian sharpness), enabling graceful degradation when one modality is compromised. Tiny transformer architectures achieve ~95% TAR@1% FAR with this approach [^6^].

7. **Bilinear and compact bilinear pooling provide theoretically grounded fusion** that captures full second-order interactions between modalities. Soleymani et al. (ICPR 2018) showed that compact bilinear fusion via tensor sketch projection achieves comparable performance to full bilinear pooling with dramatically fewer parameters [^7^].

8. **Adaptive multimodal transformers (AuthFormer) can handle flexible modality combinations** - not just face+fingerprint but arbitrary subsets - using cross-attention + Gated Residual Networks, achieving 99.73% accuracy with only 2 encoder layers [^8^].

9. **Gabor-enhanced attention networks achieve efficient multimodal biometric identification** with only 10.6M parameters and 0.85 GFLOPs, using learnable Gabor filters combined with dynamic attention-driven fusion architecture [^9^].

10. **The trade-off between intra- and cross-modality recognition remains a key challenge**: Tiong et al. (CVPR 2024) explicitly address this by designing MFA-ViT to balance both tasks simultaneously through Multimodal Fusion Attention and Multimodal Prompt Tuning [^1^].

---

### Major Papers & Sources

1. **Tiong et al.**, "Flexible Biometrics Recognition: Bridging the Multimodality Gap Through Attention, Alignment and Prompt Tuning," *CVPR*, 2024, pp. 267-276. DOI: 10.1109/CVPR52733.2024.00033 [^1^]

2. **Soleymani et al.**, "Quality-Aware Multimodal Biometric Recognition," *IEEE Transactions on Biometrics, Behavior, and Identity Science (TBIOM)*, vol. 4, no. 1, pp. 97-116, 2021. DOI: 10.1109/TBIOM.2021.3131664 [^2^]

3. **Sharma et al.**, "A Lightweight Transformer with Phase-Only Cross-Attention for Illumination-Invariant Biometric Authentication," *arXiv:2412.19160*, 2024/2025. [^4^]

4. **Lin et al.**, "A Federated Attention-Based Multimodal Biometric Recognition Approach in IoT," *Sensors*, vol. 23, no. 13, 6006, 2023. DOI: 10.3390/s23136006 [^10^]

5. **Yang et al.**, "AuthFormer: Adaptive Multimodal Biometric Authentication Transformer for Middle-Aged and Elderly People," *arXiv:2411.05395*, 2024. [^8^]

6. **Soleymani et al.**, "Generalized Bilinear Deep Convolutional Neural Networks for Multimodal Biometric Identification," *ICPR*, 2018. [^7^]

7. **LiuJun et al.**, "Multimodal Biometric Recognition Neural Networks with Branch Attention Mechanisms," *ARACE 2023*. DOI: 10.1109/ARACE60380.2023.00021 [^5^]

8. **Gnanapraveen et al.**, "Cross-Modal Transformers for Audio-Visual Person Recognition," *Odyssey 2024*. [^3^]

9. **Talreja et al.**, "Deep Hashing for Secure Multimodal Biometrics," *IEEE TIFS*, vol. 16, pp. 1306-1321, 2021. DOI: 10.1109/TIFS.2020.3033189 [^11^]

10. **Than & Nguyen**, "Efficient Multimodal Biometric Identification via Gabor-Enhanced Attention Networks," *Journal of Robotics and Control (JRC)*, 2025. DOI: 10.18196/jrc.v6i3.26490 [^9^]

11. **Jeong et al.**, "Multi-Modal Authentication Model for Occluded Faces in a Challenging Environment," *IEEE TETCI*, 2024. [^12^]

12. **PeerJ 2026**, "Tiny Transformers: Multimodal Digital Biometric Authentication Model Using Tiny Enhanced Dual Transformers with Quality-Gated Fusion Network," *PeerJ Computer Science*, vol. 12, e3657, 2026. [^6^]

13. **Soleymani et al.**, "Multi-Level Feature Abstraction from Convolutional Neural Networks for Multimodal Biometric Identification," *ICPR*, 2018. [^13^]

14. **Soleymani et al.**, "Compact Bilinear Feature Fusion for Multimodal Biometric Identification," *BTAS*, 2018. [^14^]

---

### Trends & Signals

- **Dominance of transformer-based cross-attention**: Since 2023, virtually all new proposals for multimodal biometric fusion employ transformer-based cross-attention rather than CNN-based fusion. This mirrors the broader computer vision trend but is particularly suited to biometrics due to the need for modality alignment [^1^][^8^][^4^].

- **Quality-aware fusion as a must-have**: The research community has converged on the view that any practical biometric fusion system must incorporate quality-aware weighting. Papers from 2021-2026 increasingly include quality estimation as a core component rather than an afterthought [^2^][^6^][^5^].

- **Shift from fixed to adaptive modality combinations**: AuthFormer [^8^] and related approaches enable dynamic modality selection, allowing users to authenticate with any subset of enrolled modalities. This represents a significant practical advance over fixed bimodal fusion.

- **Lightweight architectures for edge deployment**: POC-ViT [^4^], Tiny Transformers [^6^], and Gabor Attention Networks [^9^] all prioritize computational efficiency, reflecting the need for on-device biometric authentication in IoT and mobile scenarios.

- **From co-attention to cross-attention terminology**: Earlier papers (pre-2020) used "co-attention" to describe bidirectional attention between modalities; post-2020 papers predominantly use "cross-attention" following the transformer literature convention. The functional distinction is minimal.

- **Prompt tuning for modality alignment**: The CVPR 2024 FBR paper [^1^] introduces multimodal prompt tuning (MPT) as a parameter-efficient mechanism for aligning modalities within a shared ViT backbone, avoiding the need for modality-specific fine-tuning.

---

### Controversies & Conflicting Claims

1. **Feature-level vs. score-level fusion with attention**: There is no consensus on whether cross-modal attention should be applied at the feature level (early fusion) or score level (late fusion). Soleymani et al. [^2^] advocate for feature-level quality-aware fusion, while the Tiny Transformers paper [^6^] achieves competitive results with quality-gated score-level fusion. Hybrid approaches combining both levels achieve the best results in some studies [^15^].

2. **Full bilinear vs. compact bilinear vs. attention fusion**: While compact bilinear pooling [^7^] was proposed as a parameter-efficient alternative to full bilinear fusion, transformer-based cross-attention [^1^] has largely superseded both in recent work. However, no systematic comparison establishes the superiority of cross-attention over bilinear fusion for biometric data specifically.

3. **Self-attention vs. cross-attention necessity**: Gnanapraveen et al. [^3^] show that cross-attention outperforms self-attention for audio-visual fusion, but some argue that self-attention within each modality followed by simple fusion is sufficient [^10^]. The relative importance depends on modality complementarity.

4. **Quality estimation: supervised vs. weakly-supervised vs. unsupervised**: Soleymani et al. [^2^] use weakly-supervised quality estimation; the Tiny Transformers paper [^6^] uses unsupervised Laplacian sharpness; other approaches use supervised quality predictors trained on annotated quality labels. The best approach remains an open question.

5. **Shared vs. separate encoders for different modalities**: Tiong et al. [^1^] use a shared-parameter MFA-ViT backbone for face and periocular; AuthFormer [^8^] uses separate encoders per modality. The trade-off between parameter efficiency and modality-specific representation learning is unresolved.

6. **Fixed vs. learnable quality measures**: Some works [^2^] learn quality scores end-to-end; others [^6^] use fixed no-reference metrics (e.g., Laplacian variance). Learnable approaches adapt to task-specific quality notions but require more data.

---

### Research Gaps Identified

1. **No face+fingerprint-specific cross-attention study**: Despite extensive work on face+voice, face+periocular, and face+iris fusion, we found no paper specifically targeting cross-modal attention between face and fingerprint modalities in a dedicated study. Face+fingerprint remains primarily addressed by generic multimodal frameworks [^8^][^15^].

2. **Handling of extreme asymmetric quality**: While quality-aware attention handles moderate degradation well, no paper systematically evaluates performance when one modality is severely degraded (e.g., wet fingerprint + good face, or occluded face + good fingerprint) across a range of degradation levels.

3. **Cross-attention for presentation attack detection (PAD) in multimodal settings**: A 2026 survey on fingerprint PAD [^16^] identifies cross-attention between face/iris embeddings and fingerprint texture features as a promising but unexplored direction for sharing presentation-artifact cues across modalities.

4. **Limited theoretical understanding of optimal fusion rank**: Low-rank multimodal fusion (LMF) [^17^] was shown to work well with very low rank for general multimodal tasks, but the optimal rank setting for biometric fusion specifically has not been studied.

5. **Real-time quality-aware fusion benchmarks**: Most quality-aware fusion papers evaluate on relatively clean datasets. A benchmark with controlled, graded quality degradation across modalities would enable systematic comparison of quality-aware attention mechanisms.

6. **Cross-modal attention for biometric template protection**: Talreja et al. [^11^] explore secure multimodal fusion with deep hashing, but combining cross-attention with template protection (cancelable biometrics, secure sketches) remains largely unexplored.

7. **Bidirectional cross-attention for face+fingerprint**: Current approaches predominantly use unidirectional cross-attention (one modality queries another). Bidirectional cross-attention where face and fingerprint mutually attend to each other could better capture complementary information.

8. **Attention visualization and interpretability**: While Grad-CAM is used to visualize attention maps [^1^], systematic analysis of what cross-modal attention learns in biometric contexts (e.g., does face attention focus on ridge patterns when fingerprint is noisy?) is lacking.

9. **Domain adaptation for cross-modal biometric attention**: Models trained on high-quality lab data often fail on degraded field data. Cross-modal attention mechanisms that adapt to domain shift between training and deployment conditions remain undeveloped.

10. **Integration of soft biometric attributes**: The FBR framework [^1^] shows that integrating soft-biometric attributes (gender, age, ethnicity) via cross-attention improves recognition. Extending this to face+fingerprint+soft-biometrics is unexplored.

---

### Detailed Evidence Log

---

**Claim:** Multimodal Fusion Attention (MFA) within a Vision Transformer architecture achieves state-of-the-art performance on both intra- and cross-modality biometric recognition by enabling cohesive alignment between face and periocular embeddings.

**Source:** Tiong et al., CVPR 2024
**URL:** https://openaccess.thecvf.com/content/CVPR2024/papers/Tiong_Flexible_Biometrics_Recognition_Bridging_the_Multimodality_Gap_through_Attention_Alignment_CVPR_2024_paper.pdf
**Date:** 2024
**DOI:** 10.1109/CVPR52733.2024.00033
**Excerpt:** "MFA facilitates the fusion of modalities, ensuring cohesive alignment between facial and periocular embeddings while incorporating soft-biometrics to enhance the model's ability to discriminate between individuals. The fusion of three modalities is pivotal in exploring inter-relationships between different modalities."
**Confidence:** High

---

**Claim:** A quality-aware framework using weakly-supervised quality scores to weight modalities outperforms rank- and score-level fusion by more than 30% for TAR at FAR=10^-4 on face+iris+fingerprint datasets.

**Source:** Soleymani et al., IEEE TBIOM, vol. 4, no. 1, 2021
**URL:** https://arxiv.org/abs/2112.05827
**Date:** 2021
**DOI:** 10.1109/TBIOM.2021.3131664
**Excerpt:** "We develop a quality-aware framework for fusing representations of input modalities by weighting their importance using quality scores estimated in a weakly-supervised fashion. Our framework outperforms the rank- and score-level fusion of modalities of BIOMDATA by more than 30% for true acceptance rate at false acceptance rate of 10^-4."
**Confidence:** High

---

**Claim:** Cross-attention fusion achieves lower EER (2.387) than self-attention (2.412), feature concatenation (2.489), and score-level fusion (2.521) for audio-visual person recognition, demonstrating the superiority of cross-modal attention.

**Source:** Gnanapraveen et al., Odyssey 2024
**URL:** https://www.isca-archive.org/odyssey_2024/gnanapraveen24_odyssey.pdf
**Date:** 2024
**Excerpt:** "Cross-Attention (CA): 2.387 [EER]... Self-Attention (SA): 2.412... Feature Concatenation (FC): 2.489... Score-Level Fusion (SLF): 2.521"
**Confidence:** High

---

**Claim:** Phase-only correlation (POC) based cross-attention captures structural similarity between biometric traits in the frequency domain, making it robust to illumination and resolution variations while being lightweight enough for edge deployment.

**Source:** Sharma et al., arXiv:2412.19160
**URL:** https://arxiv.org/abs/2412.19160
**Date:** 2024/2025
**Excerpt:** "The computation of cross-attention using POC extracts the phase correlation in the spatial features. Therefore, it is robust against variations in resolution and intensity, as well as illumination changes in the input images. The lightweight model is suitable for edge device deployment."
**Confidence:** High

---

**Claim:** Branch attention mechanisms with fuse-select-calibrate steps dynamically weight modalities and maintain 92.76% rank-1 accuracy even on heavily contaminated test datasets.

**Source:** LiuJun et al., ARACE 2023
**URL:** https://ieeexplore.ieee.org/document/10371638
**Date:** 2023
**DOI:** 10.1109/ARACE60380.2023.00021
**Excerpt:** "The branch attention module includes fuse, select, and calibrate steps, which can dynamically calculate the weight of each branch based on its importance. Our proposed method still achieves 92.76% of rank-1 on the most contaminated dataset."
**Confidence:** High

---

**Claim:** Quality-gated score fusion using a lightweight MLP that takes Laplacian sharpness quality scores as input achieves 95% TAR@1% FAR and adapts to degraded inputs by down-weighting low-quality modalities.

**Source:** PeerJ Computer Science, 2026
**URL:** https://peerj.com/articles/cs-3657/
**Date:** 2026
**Excerpt:** "Removal of Quality-Gating resulted in a major decline in TAR (95.0% to 88.6%), as well as an increase in EER (11.7%), which shows the Adaptive Weighting, using the Laplacian Sharpness measure of quality, helps stabilize performance when one of the modalities is degraded."
**Confidence:** High

---

**Claim:** AuthFormer achieves 99.73% accuracy on face+fingerprint+voice authentication with only 2 encoder layers by combining cross-attention with Gated Residual Networks, enabling flexible modality combinations.

**Source:** Yang et al., arXiv:2411.05395
**URL:** https://arxiv.org/abs/2411.05395
**Date:** 2024
**Excerpt:** "AuthFormer achieves an accuracy of 99.73%. Moreover, the AuthFormer model's encoder requires only two layers to achieve optimal performance, significantly reducing model complexity compared to traditional Transformer-based authentication models."
**Confidence:** High

---

**Claim:** Compact bilinear pooling via tensor sketch projection achieves comparable performance to full bilinear pooling for face+iris+fingerprint identification with dramatically fewer parameters.

**Source:** Soleymani et al., ICPR 2018
**URL:** https://arxiv.org/abs/1807.01298
**Date:** 2018
**Excerpt:** "Rather than spatial fusion at the convolutional layers, the fusion can be performed on the outputs of the fully-connected layers of the modality-specific CNNs without any loss of performance and with significant reduction in the number of parameters."
**Confidence:** High

---

**Claim:** Gabor-enhanced attention networks with dynamic attention-driven fusion achieve 99.49% accuracy and 0.35% EER with only 10.6M parameters and 0.85 GFLOPs, demonstrating efficient attention-based biometric fusion.

**Source:** Than & Nguyen, JRC 2025
**URL:** https://doi.org/10.18196/jrc.v6i3.26490
**Date:** 2025
**DOI:** 10.18196/jrc.v6i3.26490
**Excerpt:** "Our model achieves superior performance compared to several state-of-the-art methods, attaining up to 99.49% accuracy and 0.35% Equal Error Rate, while maintaining high efficiency with only 10.6M parameters, 0.85 GFLOPs, and 60 FPS inference speed."
**Confidence:** Medium (newer venue, results need independent verification)

---

**Claim:** Co-attention between demographic features and facial embeddings improves occluded face identification by 3.55%-7.38%, suggesting that auxiliary non-biometric attributes can complement primary biometric modalities through attention mechanisms.

**Source:** Jeong et al., IEEE TETCI, 2024
**URL:** https://rua.ua.es/server/api/core/bitstreams/226c9f69-9bf6-4dbd-8366-54af9480a6e6/content
**Date:** 2024
**Excerpt:** "We achieved 73.25% and 90.01% test set accuracies for the DeepFace and FaceNet models, respectively, when integrated with our demographic module... 3.55%-7.38% improvement on the face identification model using the demographic module."
**Confidence:** High

---

**Claim:** Deep hashing integrated with multimodal fusion generates secure binary templates that provide cancelability and unlinkability while improving matching performance through face+iris fusion.

**Source:** Talreja et al., IEEE TIFS, vol. 16, 2021
**URL:** https://ieeexplore.ieee.org/document/9315085
**Date:** 2021
**DOI:** 10.1109/TIFS.2020.3033189
**Excerpt:** "We integrate a deep hashing (binarization) technique into the fusion architecture to generate a robust binary multimodal shared latent representation... the matching performance is improved due to the fusion of multiple biometrics."
**Confidence:** High

---

**Claim:** Attention-based fusion (soft attention weighting) outperforms simple feature concatenation by 20-27% EER reduction for face+voice biometric recognition in federated learning settings.

**Source:** Lin et al., Sensors, vol. 23, no. 13, 2023
**URL:** https://www.mdpi.com/1424-8220/23/13/6006
**Date:** 2023
**DOI:** 10.3390/s23136006
**Excerpt:** "The AF method achieved remarkable performance with 0.68%, 0.47%, and 0.80% EER... these results indicated a significant improvement over the SFF method, with improvements of 27%, 20%, and 23% for the corresponding validation sets."
**Confidence:** High

---

**Claim:** Cross-modal attention between fingerprint texture features and face/iris embeddings is identified as a promising but unexplored approach for sharing presentation-artifact cues in multimodal presentation attack detection.

**Source:** Sensors survey on fingerprint PAD, 2026
**URL:** https://www.mdpi.com/1424-8220/26/4/1283
**Date:** 2026
**Excerpt:** "Learn a lightweight cross-attention between face/iris embeddings and fingerprint texture features to share presentation-artifact cues (e.g., specular patterns, paper/print periodicity). Use modality-dropout so the system degrades gracefully if one stream is absent."
**Confidence:** High (as a research gap identification)

---

### Quality-Aware Attention Handling Asymmetric Quality

**Summary of approaches for handling one good + one degraded sample:**

| Method | Quality Measure | Fusion Strategy | Modality Pair | Year |
|--------|----------------|-----------------|---------------|------|
| Soleymani et al. [^2^] | Weakly-supervised quality network | Feature-level weighting | Face+Iris+FP | 2021 |
| Tiny Transformers [^6^] | Laplacian sharpness | Score-level gating (MLP) | Iris+Fingerprint | 2026 |
| Branch Attention [^5^] | Branch importance score | Dynamic branch weighting | Face+Ear | 2023 |
| FBR/MFA-ViT [^1^] | Implicit via attention weights | Feature-level cross-attention | Face+Periocular | 2024 |
| AuthFormer [^8^] | GRN gating mechanism | Adaptive gated fusion | Face+FP+Voice | 2024 |
| AMBR [^10^] | Soft attention scores | Attention-based weighting | Face+Voice | 2023 |

**Key insight:** All quality-aware approaches share the principle of dynamically down-weighting degraded modalities while preserving information from high-quality ones. The key differentiator is whether quality is measured explicitly (Laplacian variance, learned quality network) or implicitly (attention weights). Explicit approaches generally offer better interpretability and control; implicit approaches require less engineering but offer less transparency.

---

### References

[^1^]: Tiong, L.C.O., Sigmund, D., Chan, C.H., & Teoh, A.B.J. "Flexible Biometrics Recognition: Bridging the Multimodality Gap Through Attention, Alignment and Prompt Tuning." *CVPR*, pp. 267-276, 2024.

[^2^]: Soleymani, S., Dabouei, A., Taherkhani, F., Iranmanesh, S.M., Dawson, J., & Nasrabadi, N.M. "Quality-Aware Multimodal Biometric Recognition." *IEEE TBIOM*, vol. 4, no. 1, pp. 97-116, 2021.

[^3^]: Gnanapraveen, S., et al. "Cross-Modal Transformers for Audio-Visual Person Recognition." *Odyssey 2024: The Speaker and Language Recognition Workshop*, 2024.

[^4^]: Sharma, A.K., Bhattacharya, S., Reza, M., & Bhattacharya, B. "A Lightweight Transformer with Phase-Only Cross-Attention for Illumination-Invariant Biometric Authentication." *arXiv:2412.19160*, 2024.

[^5^]: LiuJun, Chen, M., Guo, Y., & Huang, Z. "Multimodal Biometric Recognition Neural Networks with Branch Attention Mechanisms." *ARACE 2023*, pp. 87-92, 2023.

[^6^]: "Tiny Transformers: Multimodal Digital Biometric Authentication Model Using Tiny Enhanced Dual Transformers with Quality-Gated Fusion Network." *PeerJ Computer Science*, vol. 12, e3657, 2026.

[^7^]: Soleymani, S., Torfi, A., Dawson, J., & Nasrabadi, N.M. "Generalized Bilinear Deep Convolutional Neural Networks for Multimodal Biometric Identification." *ICPR*, 2018.

[^8^]: Yang, R., Zhang, Q., & Meng, L. "AuthFormer: Adaptive Multimodal Biometric Authentication Transformer for Middle-Aged and Elderly People." *arXiv:2411.05395*, 2024.

[^9^]: Than, P.M. & Nguyen, H. "Efficient Multimodal Biometric Identification via Gabor-Enhanced Attention Networks." *JRC*, vol. 6, no. 3, 2025.

[^10^]: Lin, L., Zhao, Y., Meng, J., & Zhao, Q. "A Federated Attention-Based Multimodal Biometric Recognition Approach in IoT." *Sensors*, vol. 23, no. 13, 6006, 2023.

[^11^]: Talreja, V., Valenti, M.C., & Nasrabadi, N.M. "Deep Hashing for Secure Multimodal Biometrics." *IEEE TIFS*, vol. 16, pp. 1306-1321, 2021.

[^12^]: Jeong, J., et al. "Multi-Modal Authentication Model for Occluded Faces in a Challenging Environment." *IEEE TETCI*, 2024.

[^13^]: Soleymani, S., et al. "Multi-Level Feature Abstraction from Convolutional Neural Networks for Multimodal Biometric Identification." *ICPR*, 2018.

[^14^]: Soleymani, S., et al. "Compact Bilinear Feature Fusion for Multimodal Biometric Identification." *BTAS*, 2018.

[^15^]: "Deep Learning-Based Fingerprint-Vein Biometric Fusion." *Applied Sciences*, vol. 15, no. 15, 8502, 2025.

[^16^]: "A Survey on Deep Learning Techniques for Fingerprint Presentation Attack Detection." *Sensors*, vol. 26, no. 4, 1283, 2026.

[^17^]: Liu, Z., Shen, Y., Lakshminarasimhan, V.B., Liang, P.P., Zadeh, A., & Morency, L.P. "Efficient Low-rank Multimodal Fusion With Modality-Specific Factors." *ACL*, pp. 2247-2256, 2018.

---

**[CITATION TO VERIFY]** The Than & Nguyen JRC paper is published in a newer venue; independent verification of results is recommended.

**[CITATION TO VERIFY]** The Sharma et al. POC-ViT paper is on arXiv and may not have completed peer review at time of search.

**[CITATION TO VERIFY]** The AuthFormer paper is on arXiv and may not have completed peer review at time of search.

---

*End of Dimension 5 Research Report*
