## Dimension 8: Missing Modality Handling in Multimodal Deep Learning

**Date:** 2026-01-15
**Focus:** Approaches handling missing modalities at inference time in multimodal deep learning, with focus on biometrics (face + fingerprint), 2020-2026
**Total Papers Found:** 25+

---

### Key Findings

- **Missing modality handling is a well-studied problem in medical imaging** (brain tumor segmentation) but remains **critically under-addressed in multimodal biometrics**, where most fusion papers assume all modalities are present at all times [^1^].

- **Modality dropout during training** (a.k.a. ModDrop) is the foundational technique for robustness to missing modalities, first introduced by Neverova et al. 2015 and widely adopted since. Random dropping of modality channels during training forces the network to learn cross-modal correlations while preserving modality-specific representations [^2^].

- **Multimodal Transformers are NOT naturally robust to missing modalities**. Ma et al. (CVPR 2022) conducted the first comprehensive study and found that ViLT performance drops dramatically with missing data, sometimes performing *worse* than unimodal models. The optimal fusion strategy is dataset-dependent; no universal strategy exists [^3^].

- **Learnable placeholder/proxy tokens** have emerged as a parameter-efficient solution. Cross-Modal Proxy Tokens (CMPTs) approximate the class token of a missing modality by attending to available modality tokens, requiring only a single token per modality with LoRA rank-1 adapters [^4^].

- **Prompt tuning for missing modalities** (Lee et al., CVPR 2023) enables handling incomplete modalities with less than 1% learnable parameters. Missing-aware prompts are plugged into multimodal transformers for different missing-modality cases [^5^].

- **SMIL** (Ma et al., AAAI 2021) pioneered Bayesian meta-regularization for severely missing modalities (up to 90% missing ratio), handling missing patterns in training, testing, or both flexibly [^6^].

- **Parameter-efficient adaptation** via low-rank modulation (Reza et al., 2023/2024) can bridge performance gaps from missing modalities using fewer than 1% additional parameters, outperforming dedicated networks trained for specific modality combinations [^7^].

- **ActionMAE** (Woo et al., AAAI 2023) proposed modular autoencoder-based missing modality predictive coding for action recognition, finding that transformer-based fusion is more robust to missing modalities than summation or concatenation [^8^].

- **Cross-modal knowledge distillation** (Wang et al., MICCAI 2023) adaptively identifies important modalities and distills knowledge to student modalities, enabling them to perform well even when teacher modalities are missing [^9^].

- **Arbitrary-modal inference** (Zhang et al., CVPR 2023) through CMNeXt enables semantic segmentation with any subset of modalities (RGB, Depth, LiDAR, Event), scaling from 1 to 80+ modalities with negligible parameter overhead (~0.01M per modality) [^10^].

- **In biometrics specifically**, most multimodal face+fingerprint fusion papers (e.g., Singh et al. 2020, Shen et al. 2025) use standard fusion but do NOT explicitly test or handle missing modalities at inference time. This is a critical gap [^11^].

- **Modality dropout in biometrics** has been explored with spiking neural networks (Shen et al., 2025), introducing Multimodal Adaptive Dropout (MAD) that dynamically discards features from converged/dominant modalities during training. However, this targets modality *imbalance* rather than missing-at-inference robustness [^12^].

- **Biometric fallback strategies** exist at the system level (score-level fusion with weighted averaging when one modality fails), but these are typically engineered solutions rather than learned deep learning approaches [^13^].

- **A key controversy** exists between generation-based approaches (reconstructing missing modalities) and adaptation-based approaches (learning to compensate without reconstruction). The former requires complex generative models and may introduce artifacts; the latter may not fully recover performance [^14^].

---

### Major Papers & Sources

#### Foundational/Survey Papers

1. **Wu, R., Wang, H., Chen, H.T., & Carneiro, G.** "Deep Multimodal Learning with Missing Modality: A Survey." *Transactions on Machine Learning Research*, 2026. https://openreview.net/forum?id=tc7RFcx4hT [^1^]
   - First comprehensive survey covering taxonomy: data processing (imputation vs. representation-focused) and strategy design (architecture-focused vs. optimization-focused)
   - Covers 2015-2025, but NO biometric-specific content

2. **Neverova, N., Wolf, C., Taylor, G., & Nebout, F.** "ModDrop: Adaptive Multi-modal Gesture Recognition." *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, 38(8):1692-1706, 2016/2015. DOI: 10.1109/TPAMI.2015.2461544 [^2^]
   - Introduced ModDrop: random dropping of separate modality channels during training
   - Ensures robustness to missing signals in one or several channels
   - Produces meaningful predictions from ANY number of available modalities

#### Transformer Robustness & Fusion Strategies

3. **Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X.** "Are Multimodal Transformers Robust to Missing Modality?" *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 18177-18186, 2022. DOI: 10.1109/CVPR52688.2022.01764 [^3^]
   - First systematic study of Transformer robustness to missing-modal data
   - Key finding: multimodal performance can be WORSE than unimodal when modalities are severely missing
   - Optimal fusion strategy is dataset-dependent; proposed multi-task optimization + differentiable search

4. **Ma, M., Ren, J., Zhao, L., Tulyakov, S., Wu, C., & Peng, X.** "SMIL: Multimodal Learning with Severely Missing Modality." *Proc. AAAI Conference on Artificial Intelligence*, vol. 35, pp. 2302-2310, 2021. DOI: 10.1609/aaai.v35i3.16330 [^6^]
   - Handles up to 90% missing ratio during training, testing, or both
   - Bayesian meta-regularization to generate missing modality representations
   - Uses learned cross-modal translation between modalities

#### Prompt Tuning & Parameter-Efficient Methods

5. **Lee, Y.-L., Tsai, Y.-H., Chiu, W.-C., & Lee, C.-Y.** "Multimodal Prompting With Missing Modalities for Visual Recognition." *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 14943-14952, 2023. [^5^]
   - Missing-aware prompts plugged into multimodal transformers
   - Less than 1% learnable parameters compared to full model fine-tuning
   - Handles general missing-modality cases at both training and test time
   - Code: https://github.com/yllee1009/multimodal_prompting

6. **Reza, M.K., Patil, A., Solh, M., & Asif, M.S.** "Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation." *arXiv:2310.03986*, 2023. [^7^]
   - Low-rank adaptation + modulation of intermediate features
   - Fewer than 1% additional parameters; applicable to wide range of tasks
   - Outperforms dedicated networks trained for specific modality combinations
   - Project page: https://csiplab.github.io/Missing-Modality-Adaptation/

7. **Shi, R. et al.** "MoRA: LoRA Guided Multi-Modal Disease Diagnosis with Missing Modalities." *MICCAI 2024*. [^15^]
   - Modality-aware Low-rank Adaptation with distinct up-projections per modality
   - Only 1.6% of total model parameters need training
   - Specifically designed for medical diagnosis with incomplete modalities

#### Cross-Modal Generation & Knowledge Distillation

8. **Wang, H., Ma, C., Zhang, J., Zhang, Y., Avery, J., Hull, L., & Carneiro, G.** "Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality." *MICCAI 2023*. DOI: 10.1007/978-3-031-43901-8_21 [^9^]
   - Teacher election procedure selects best-performing modalities as teachers
   - Cross-modal knowledge distillation to student modalities
   - 3.61-5.99% improvement on BraTS2018 brain tumor segmentation

9. **Hoffman, J., Gupta, S., & Darrell, T.** "Learning with Side Information through Modality Hallucination." *CVPR 2016*. [^16^]
   - Hallucination Network (HN) predicts missing depth features from RGB
   - Aligns intermediate layer features using L2 loss
   - Extended to pose estimation and land cover classification

#### Proxy Tokens & Mask Tokens

10. **Reza, M.K., Patil, A., Solh, M., & Asif, M.S.** "Robust Multimodal Learning via Cross-Modal Proxy Tokens." *arXiv:2501.17823*, 2025. [^4^]
    - Cross-Modal Proxy Tokens (CMPTs) approximate missing modality class tokens
    - Single token per modality with LoRA rank-1 adapters
    - Gating module substitutes missing class token with CMPT from available modality
    - Outperforms 12 recent baselines across 4 datasets

11. **Reza, M.K., Nezakati, N., Patil, A., Solh, M., & Asif, M.S.** "U2A: Unified Unimodal Adaptation for Robust and Efficient Multimodal Learning." *arXiv:2501.01120*, 2025. [^17^]
    - Mask Tokens (MT) generate missing modality features from available modalities
    - Jointly fine-tunes pretrained unimodal encoders using LoRA
    - Eliminates need for complex training strategies (alternating, gradient modification)

#### Arbitrary-Modal Inference

12. **Zhang, J., Liu, R., Shi, H., Yang, K., Reiß, S., Peng, K., Fu, H., Wang, K., & Stiefelhagen, R.** "Delivering Arbitrary-Modal Semantic Segmentation." *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. [^10^]
    - CMNeXt model with Self-Query Hub (SQ-Hub) for arbitrary-modal fusion
    - Scales from 1 to 81 modalities with ~0.01M parameters per modality
    - DeLiVER benchmark includes sensor failure cases
    - Code: https://github.com/jamycheung/DELIVER

#### Action Recognition & Other Domains

13. **Woo, S., Lee, S., Park, Y., Nugroho, M.A., & Kim, C.** "Towards Good Practices for Missing Modality Robust Action Recognition." *Proc. AAAI Conference on Artificial Intelligence*, vol. 37, pp. 2776-2784, 2023. [^8^]
    - ActionMAE: modular autoencoder reconstructs missing modality tokens
    - Transformer-based fusion more robust than summation or concatenation
    - Achieves state-of-the-art on multiple benchmarks with missing modalities

14. **Zhang, J. et al.** "Multimodal Representation Learning by Alternating Unimodal Adaptation." *CVPR 2024*. [^18^]
    - MLA: reframes joint multimodal learning as alternating unimodal learning
    - Test-time uncertainty-based model fusion mechanism
    - Handles both complete and missing modality scenarios

#### Medical Image Segmentation (Missing Modalities)

15. **Zhang, Y. et al.** "mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation." *MICCAI 2022*. [^19^]
    - First Transformer for brain tumor segmentation robust to ANY subset of modalities
    - Hybrid modality-specific encoders + inter-modal Transformer
    - 19.07% average Dice improvement with only one available modality

16. **Chartsias, A. et al.** "HeMIS: Hetero-Modal Image Segmentation." *MICCAI 2016*. [^20^]
    - Early work on heterogeneous multimodal segmentation with missing modalities
    - Uses mean filling and imputation MLP baselines
    - Foundation for many follow-up works in medical imaging

#### Biometric-Specific Papers

17. **Shen, Y., Yang, X., Liu, X., Wan, J., & Xia, N.** "Towards Energy-Effective Multimodal Biometric Recognition Via Information Bottleneck Fusion Spiking Neural Networks." *Journal of Advanced Research in Computer Science and Engineering*, 2025. [^12^]
    - Multimodal Adaptive Dropout (MAD) for biometric modality balancing
    - Information Bottleneck fusion to reduce redundancy
    - Tested on CASIA, Iris-Fingerprint, NUPT-FPV datasets
    - Evaluates performance under modality loss conditions (Table 3)

18. **El-Bousty, M. et al.** "Multimodal Biometric Authentication: A Novel Deep Learning Framework Integrating ECG, Fingerprint, and Finger Knuckle Print." *IOP Science*, 2025. DOI: 10.1088/2631-8695/ad9aa0 [^13^]
    - Explicit fallback strategies for single- and dual-modality authentication
    - System maintains high accuracy even when one or two modalities are compromised
    - One of the few biometric papers explicitly addressing partial modality failure

19. **Korse, N.N. et al.** "Context-Aware Decision Fusion for Multimodal Access Control Under Contradictory Biometric Evidence." *MDPI Computers*, 15(4):208, 2026. [^21^]
    - Decision-level fusion with reliability gating and contradiction detection
    - Handles availability failure scenarios (one or more modalities absent)
    - Experimental protocol includes S9 (availability failure) and S10 (multi-failure) cases
    - Uses face (VGGFace2), voice (VoxCeleb2), fingerprint (FVC2004)

#### Modality Dropout & Training Strategies

20. **Furbock, S. et al.** "Learning Contrastive Multimodal Fusion with Improved Modality Dropout." *MICCAI 2025*. [^22^]
    - Learnable modality tokens replace fixed zero matrices in modality dropout
    - Simultaneous modality dropout supervises all modality combinations without sampling
    - Inter-modality contrastive learning

---

### Trends & Signals

- **Trend 1: From generation to adaptation.** Early methods focused on reconstructing missing modalities (GANs, autoencoders). Recent approaches (2023-2025) favor parameter-efficient adaptation (prompts, LoRA, proxy tokens) that avoid expensive generation [^1^][^4^][^7^].

- **Trend 2: Parameter efficiency.** The field is moving toward methods that require <1% additional parameters to handle missing modalities, making them practical for deployment [^5^][^7^][^15^].

- **Trend 3: Arbitrary-modal inference.** Rather than handling specific missing modality cases, newer methods aim to work with ANY subset of modalities. CMNeXt scales to 81 modalities [^10^]; mmFormer handles any subset of 4 MRI modalities [^19^].

- **Trend 4: Biometrics lags behind.** While medical imaging, autonomous driving, and action recognition have extensive missing-modality research, multimodal biometric fusion (face+fingerprint) papers overwhelmingly assume all modalities are present. This is a critical research gap [^11^].

- **Trend 5: Modality-aware training.** Adaptive/learning-based modality dropout (rather than random dropout) is emerging as a way to balance modality learning and improve robustness [^12^][^22^].

- **Trend 6: Cross-modal proxy approaches.** Using available modalities to "hallucinate" or proxy missing modality representations is gaining traction over explicit generation [^4^][^16^][^17^].

---

### Controversies & Conflicting Claims

1. **Generation vs. Adaptation:** Generation-based methods (ActionMAE, GAN-based imputation) claim better performance by reconstructing missing modalities, but require significantly more computation and may introduce artifacts. Adaptation-based methods (prompt tuning, LoRA) are more efficient but may not fully recover performance [^14^]. CMPTs represent a middle ground [^4^].

2. **Is modality dropout sufficient?** Modality dropout alone improves robustness but does NOT fully recover performance under severe missing conditions. The CMPT paper shows modality dropout improves over baseline but still has significant gaps compared to using proxy tokens [^4^].

3. **Optimal fusion strategy:** Ma et al. (CVPR 2022) found that optimal fusion strategy is dataset-dependent [^3^], while Woo et al. (AAAI 2023) found transformer-based fusion generally more robust than summation/concatenation [^8^]. These findings are context-dependent.

4. **Biometric fusion assumes completeness:** Most face+fingerprint fusion papers (e.g., Singh et al. 2020) evaluate only on complete modalities and report accuracy/EER under this assumption. The Shen et al. 2025 paper is one of the few that evaluates under modality loss conditions [^11^][^12^].

5. **Negative co-learning:** Magal et al. (2025) showed that aggressive dropout can convert negative co-learning (where multimodal training hurts unimodal performance) into positive co-learning with up to 20% accuracy gains in some scenarios [^23^].

---

### Research Gaps Identified

1. **CRITICAL GAP: Missing modality handling in face+fingerprint biometrics.** The vast majority of multimodal biometric fusion papers assume both modalities are always present at inference. There is virtually no work on deep learning-based missing modality compensation specifically for face+fingerprint fusion. Score-level fallback is used as an engineering workaround [^11^].

2. **No benchmark for missing-modality biometric evaluation.** Existing biometric datasets (e.g., FVC2004, CASIA) do not include protocols for evaluating missing modality scenarios. Researchers must manually simulate sensor failures [^13^].

3. **Generation quality for biometric modalities.** Reconstructing a missing fingerprint or face image from the other modality is significantly harder than reconstructing missing depth from RGB, due to the weak cross-modal correlation between face and fingerprint (different sensors, different biological characteristics) [^14^].

4. **Security implications of missing modality compensation.** If an attacker knows the system compensates for missing face with fingerprint features (or vice versa), they may target the "weaker" modality. No work has studied adversarial robustness of missing-modality compensation in biometrics [^21^].

5. **Real-world sensor failure patterns.** Most missing modality research assumes random or uniform missing patterns. In practice, biometric sensor failures follow specific patterns (e.g., fingerprint sensor dry skin, face camera occlusion, lighting changes). Domain-specific failure modeling is needed [^8^].

6. **Trade-off between completeness and efficiency.** Methods like CMNeXt [^10^] that handle arbitrary modalities add complexity. For resource-constrained biometric devices, simpler parameter-efficient methods [^7^] may be more practical.

7. **Cross-dataset generalization.** Most missing modality methods are evaluated on a single dataset. The ability to generalize to unseen missing rates or different modality combinations across datasets is understudied [^4^].

8. **Theoretical understanding.** There is limited theoretical analysis of WHY certain modalities are more important, WHAT the performance lower bound is when modalities are missing, and HOW to design training strategies that encourage uniform reliance on modalities [^4^].

---

### Detailed Evidence Log

#### Claim 1: Modality dropout is foundational for missing-modality robustness
**Claim:** Random dropping of modality channels during training (ModDrop) ensures robustness to missing signals at inference time, producing meaningful predictions from any number of available modalities.
**Source:** IEEE T-PAMI
**URL:** https://chriswolfvision.github.io/www/papers/pami2015.pdf
**Date:** 2016
**Excerpt:** "The proposed ModDrop training technique ensures robustness of the classifier to missing signals in one or several channels to produce meaningful predictions from any number of available modalities."
**Confidence:** HIGH

#### Claim 2: Multimodal Transformers are NOT naturally robust to missing modalities
**Claim:** Transformer models (e.g., ViLT) degrade dramatically with missing-modal data, sometimes performing worse than unimodal models. Optimal fusion strategy is dataset-dependent.
**Source:** CVPR 2022
**URL:** https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html
**Date:** 2022
**Excerpt:** "We empirically find that Transformer models are sensitive to missing-modal data... the multimodal performance is even worse than the unimodal one on MM-IMDb and UPMC Food-101."
**Confidence:** HIGH

#### Claim 3: Learnable proxy tokens outperform modality dropout alone
**Claim:** Cross-Modal Proxy Tokens with LoRA rank-1 adapters achieve state-of-the-art performance across missing modality scenarios, improving 4-9% over prompt-based approaches.
**Source:** arXiv 2025
**URL:** https://arxiv.org/abs/2501.17823
**Date:** 2025
**Excerpt:** "CMPTs achieve a 4.18%-9.40% improvement in F1-macro score on MM-IMDb dataset compared to the most recent approach by Kim & Kim (2024)."
**Confidence:** HIGH

#### Claim 4: Prompt tuning handles missing modalities with <1% parameters
**Claim:** Missing-aware prompts plugged into multimodal transformers handle general missing-modality cases while requiring less than 1% learnable parameters.
**Source:** CVPR 2023
**URL:** https://openaccess.thecvf.com/content/CVPR2023/html/Lee_Multimodal_Prompting_With_Missing_Modalities_for_Visual_Recognition_CVPR_2023_paper.html
**Date:** 2023
**Excerpt:** "Our modality-missing-aware prompts can be plugged into multimodal transformers to handle general missing-modality cases, while only requiring less than 1% learnable parameters."
**Confidence:** HIGH

#### Claim 5: Arbitrary-modal inference scales to 81 modalities
**Claim:** CMNeXt enables semantic segmentation with any subset of modalities, scaling from 1 to 81 modalities with ~0.01M parameters per additional modality.
**Source:** CVPR 2023
**URL:** https://arxiv.org/abs/2303.01480
**Date:** 2023
**Excerpt:** "Our CMNeXt achieves state-of-the-art performance... allowing to scale from 1 to 81 modalities on the DeLiVER, KITTI-360, MFNet, NYU Depth V2, UrbanLF, and MCubeS datasets."
**Confidence:** HIGH

#### Claim 6: SMIL handles severely missing modalities (up to 90% missing)
**Claim:** SMIL uses Bayesian meta-regularization to handle up to 90% missing modality ratio during training, testing, or both, generating missing modality representations flexibly.
**Source:** AAAI 2021
**URL:** https://ojs.aaai.org/index.php/AAAI/article/view/16330
**Date:** 2021
**Excerpt:** "We consider an even more challenging setting that the missing ratio can be as much as 90%... our model to uniformly tackle three different missing patterns in training, testing, or both."
**Confidence:** HIGH

#### Claim 7: Biometric fusion papers ASSUME both modalities present (CRITICAL GAP)
**Claim:** Most multimodal biometric face+fingerprint fusion papers do NOT evaluate or handle missing modality scenarios; they assume all modalities are always present.
**Source:** Multiple sources (Singh et al. 2020 IJISMD, Shen et al. 2025, El-Bousty et al. 2025)
**URL:** https://iopscience.iop.org/article/10.1088/2631-8695/ad9aa0
**Date:** 2020-2025
**Excerpt:** From Singh et al.: "The proposed multimodal biometrics system presents recognition based on face detection and fingerprint physiological traits" - no mention of missing modality handling. From El-Bousty et al.: "We evaluated the system's performance under conditions of partial modality failure, implementing fallback strategies" - one of the few exceptions.
**Confidence:** HIGH

#### Claim 8: Cross-modal knowledge distillation identifies and transfers from important modalities
**Claim:** LCKD adaptively identifies important modalities via teacher election and distills knowledge to help other modalities, achieving 3.61-5.99% improvement on BraTS2018.
**Source:** MICCAI 2023
**URL:** https://arxiv.org/abs/2310.01035
**Date:** 2023
**Excerpt:** "LCKD outperforms other methods by a considerable margin, improving the state-of-the-art performance by 3.61% for enhancing tumour, 5.99% for tumour core, and 3.76% for whole tumour."
**Confidence:** HIGH

#### Claim 9: Parameter-efficient adaptation bridges performance gap
**Claim:** Simple linear transformations (scaling, shifting, low-rank) can adapt pretrained multimodal networks to arbitrary modality combinations with less than 0.7% parameters.
**Source:** arXiv 2023/2024
**URL:** https://arxiv.org/abs/2310.03986
**Date:** 2023
**Excerpt:** "The proposed adaptation requires extremely small number of parameters (e.g., fewer than 0.7% of the total parameters) and applicable to a wide range of modality combinations and tasks."
**Confidence:** HIGH

#### Claim 10: ActionMAE reconstructs missing modality features
**Claim:** Modular autoencoder-based missing modality predictive coding achieves SOTA on action recognition benchmarks while maintaining competitive performance with missing modalities.
**Source:** AAAI 2023
**URL:** https://ojs.aaai.org/index.php/AAAI/article/view/25378
**Date:** 2023
**Excerpt:** "We propose a simple modular network, ActionMAE, which learns missing modality predictive coding by randomly dropping modality features and tries to reconstruct them with the remaining modality features."
**Confidence:** HIGH

#### Claim 11: Biometric systems can implement fallback at score level
**Claim:** Multimodal biometric systems can maintain high accuracy under partial modality failure by implementing fallback strategies for single- and dual-modality authentication at the score fusion level.
**Source:** IOP Science 2025
**URL:** https://iopscience.iop.org/article/10.1088/2631-8695/ad9aa0
**Date:** 2025
**Excerpt:** "The results demonstrated that even when one or two modalities were compromised, the system maintained high accuracy and reliability."
**Confidence:** MEDIUM (score-level fusion, not deep learning-based compensation)

#### Claim 12: Adaptive modality dropout balances biometric modality learning
**Claim:** Multimodal Adaptive Dropout (MAD) dynamically discards features from converged/dominant modalities during training, mitigating imbalance and improving robustness on biometric datasets.
**Source:** Journal of Advanced Research in Computer Science and Engineering 2025
**URL:** https://www.opastpublishers.com/open-access-articles/towards-energyeffective-multimodal-biometric-recognition-via-information-bottleneck-fusion-spiking-neural-networks-9369.html
**Date:** 2025
**Excerpt:** "MAD detects the convergence status of modalities during training and applies random dropout to features of converged and dominant modalities to reduce feature redundancy."
**Confidence:** MEDIUM (evaluated on modality balance, not missing-at-inference)

#### Claim 13: Comprehensive survey reveals taxonomy of approaches
**Claim:** Deep multimodal learning with missing modality approaches can be categorized into: (1) data processing (modality imputation vs. representation-focused) and (2) strategy design (architecture-focused vs. optimization-focused).
**Source:** TMLR 2026
**URL:** https://arxiv.org/abs/2409.07825
**Date:** 2024/2026
**Excerpt:** "We review current deep MLMM methods from two key aspects: data processing and strategy design, based on the contributions of existing work."
**Confidence:** HIGH

#### Claim 14: Context-aware decision fusion handles biometric sensor failure
**Claim:** Decision-level fusion with reliability gating and contradiction detection can handle availability failure (one or more modalities absent) in multimodal biometric access control.
**Source:** MDPI Computers 2026
**URL:** https://www.mdpi.com/2073-431X/15/4/208
**Date:** 2026
**Excerpt:** "Each modality-specific decision agent operates independently... This design choice preserves modality-agnosticism and facilitates integration with heterogeneous biometric systems."
**Confidence:** MEDIUM (decision-level, not deep learning feature-level)

---

### Summary Table: Methods for Missing Modality Handling

| Method | Year | Venue | Approach | Modality Dropout | Proxy/Token | Arbitrary-Subset | Biometric-Specific |
|--------|------|-------|----------|------------------|-------------|-----------------|-------------------|
| ModDrop [^2^] | 2015/2016 | T-PAMI | Random modality dropout | Yes | No | Yes | No |
| HeMIS [^20^] | 2016 | MICCAI | Hetero-modal segmentation | Partial | No | Yes | No |
| Hoffman et al. [^16^] | 2016 | CVPR | Hallucination Network | No | No | No | No |
| SMIL [^6^] | 2021 | AAAI | Bayesian meta-regularization | No | No | Yes | No |
| Ma et al. [^3^] | 2022 | CVPR | Optimal fusion search | Yes | No | No | No |
| mmFormer [^19^] | 2022 | MICCAI | Medical Transformer | No | No | Yes | No |
| ActionMAE [^8^] | 2023 | AAAI | Autoencoder reconstruction | Yes | No | Partial | No |
| Lee et al. [^5^] | 2023 | CVPR | Missing-aware prompts | Yes | Yes (prompts) | Partial | No |
| LCKD [^9^] | 2023 | MICCAI | Cross-modal distillation | No | No | No | No |
| Reza et al. [^7^] | 2023 | arXiv | LoRA adaptation | Yes | No | Yes | No |
| Zhang et al. [^10^] | 2023 | CVPR | CMNeXt arbitrary-modal | No | No | **Yes** | No |
| MoRA [^15^] | 2024 | MICCAI | Modality-aware LoRA | No | No | Partial | No |
| CMPT [^4^] | 2025 | arXiv | Cross-modal proxy tokens | Yes | **Yes** | Partial | No |
| U2A [^17^] | 2025 | arXiv | Mask tokens + LoRA | Yes | **Yes** | Partial | No |
| Shen et al. [^12^] | 2025 | Journal | Adaptive dropout + SNN | **Yes** | No | No | **Yes** |
| El-Bousty [^13^] | 2025 | IOP | Score-level fallback | No | No | Partial | **Yes** |
| Korse et al. [^21^] | 2026 | MDPI | Decision fusion | No | No | Partial | **Yes** |

### Key Observation

**Only 3 out of 17 methods (Shen et al. 2025, El-Bousty 2025, Korse et al. 2026) are biometric-specific, and ALL of these use non-deep-learning approaches (adaptive dropout, score-level fallback, or decision-level fusion) for handling missing modalities. NONE of the state-of-the-art deep learning methods for missing modality handling (CMPT, U2A, prompt tuning, etc.) have been applied to multimodal face+fingerprint biometrics.**

This represents a **critical research gap** with significant practical implications: in real-world biometric systems, fingerprint sensors may fail due to dry skin, cuts, or dirt; face cameras may fail due to occlusion, extreme lighting, or pose. Current deep learning-based biometric fusion does not handle these scenarios robustly.

---

### References (Cited by Number)

[^1^]: Wu, R., Wang, H., Chen, H.T., & Carneiro, G. "Deep Multimodal Learning with Missing Modality: A Survey." TMLR, 2026.
[^2^]: Neverova, N., Wolf, C., Taylor, G., & Nebout, F. "ModDrop: Adaptive Multi-modal Gesture Recognition." IEEE T-PAMI, 38(8):1692-1706, 2016.
[^3^]: Ma, M., Ren, J., Zhao, L., Testuggine, D., & Peng, X. "Are Multimodal Transformers Robust to Missing Modality?" CVPR, pp. 18177-18186, 2022.
[^4^]: Reza, M.K., Patil, A., Solh, M., & Asif, M.S. "Robust Multimodal Learning via Cross-Modal Proxy Tokens." arXiv:2501.17823, 2025.
[^5^]: Lee, Y.-L., Tsai, Y.-H., Chiu, W.-C., & Lee, C.-Y. "Multimodal Prompting With Missing Modalities for Visual Recognition." CVPR, pp. 14943-14952, 2023.
[^6^]: Ma, M., Ren, J., Zhao, L., Tulyakov, S., Wu, C., & Peng, X. "SMIL: Multimodal Learning with Severely Missing Modality." AAAI, vol. 35, pp. 2302-2310, 2021.
[^7^]: Reza, M.K., Patil, A., Solh, M., & Asif, M.S. "Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation." arXiv:2310.03986, 2023.
[^8^]: Woo, S., Lee, S., Park, Y., Nugroho, M.A., & Kim, C. "Towards Good Practices for Missing Modality Robust Action Recognition." AAAI, vol. 37, pp. 2776-2784, 2023.
[^9^]: Wang, H., Ma, C., Zhang, J., et al. "Learnable Cross-modal Knowledge Distillation for Multi-modal Learning with Missing Modality." MICCAI 2023.
[^10^]: Zhang, J., Liu, R., Shi, H., Yang, K., et al. "Delivering Arbitrary-Modal Semantic Segmentation." CVPR, 2023.
[^11^]: Singh, L.K., Khanna, M., & Garg, H. "Multimodal Biometric Based on Fusion of Ridge Features with Minutiae Features and Face Features." IJISMD, vol. 11(1), pp. 37-57, 2020.
[^12^]: Shen, Y., Yang, X., Liu, X., Wan, J., & Xia, N. "Towards Energy-Effective Multimodal Biometric Recognition Via Information Bottleneck Fusion Spiking Neural Networks." 2025.
[^13^]: El-Bousty, M. et al. "Multimodal Biometric Authentication: Integrating ECG, Fingerprint, and Finger Knuckle Print." IOP Science, 2025.
[^14^]: From comparison of generation-based (ActionMAE, SMIL) vs. adaptation-based (CMPT, prompt tuning) approaches in the literature.
[^15^]: Shi, R. et al. "MoRA: LoRA Guided Multi-Modal Disease Diagnosis with Missing Modalities." MICCAI 2024.
[^16^]: Hoffman, J., Gupta, S., & Darrell, T. "Learning with Side Information through Modality Hallucination." CVPR 2016.
[^17^]: Reza, M.K., Nezakati, N., Patil, A., Solh, M., & Asif, M.S. "U2A: Unified Unimodal Adaptation for Robust and Efficient Multimodal Learning." arXiv:2501.01120, 2025.
[^18^]: Zhang, J. et al. "Multimodal Representation Learning by Alternating Unimodal Adaptation." CVPR 2024.
[^19^]: Zhang, Y. et al. "mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation." MICCAI 2022.
[^20^]: Chartsias, A. et al. "HeMIS: Hetero-Modal Image Segmentation." MICCAI 2016.
[^21^]: Korse, N.N. et al. "Context-Aware Decision Fusion for Multimodal Access Control." MDPI Computers, 15(4):208, 2026.
[^22^]: Furbock, S. et al. "Learning Contrastive Multimodal Fusion with Improved Modality Dropout." MICCAI 2025.
[^23^]: Magal et al. (2025) on negative-to-positive co-learning via aggressive dropout, referenced in modality dropout surveys.
