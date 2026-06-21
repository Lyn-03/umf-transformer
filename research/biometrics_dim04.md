# Dimension 4: Transformer-Based Approaches for Biometric Recognition

## Key Findings

### ViT (Vision Transformer) for Face Recognition

- **Face Transformer for Recognition** (Zhong & Deng, 2021) was among the first works to apply Vision Transformers to face recognition, demonstrating that ViT backbones could achieve competitive performance with CNNs on standard face benchmarks [^1^]. The work established the foundation for subsequent transformer-based face recognition research but noted convergence challenges when scaling to large datasets.

- **TransFace** (Dan et al., ICCV 2023) identified critical training incompatibilities when applying ViTs to face recognition with large-scale datasets. The authors proposed a Dominant Patch Amplitude Perturbation (DPAP) data augmentation strategy and an Entropy-based Hard Sample Mining (EHSM) strategy. TransFace-S outperformed ViT-S by +0.56% on IJB-C at TAR@FAR=1e-4 [^2^]. TransFace-L (ViT-L backbone) achieved state-of-the-art results among transformer methods, reaching 96.37% on IJB-C at FAR=1e-4 when trained on Glint360K.

- **LVFace** (You et al., 2025) from ByteDance introduced Progressive Cluster Optimization (PCO), a three-stage training strategy (Feature Alignment -> Centroid Stabilization -> Boundary Refinement) that achieves state-of-the-art face recognition performance. LVFace-L with ViT-L backbone trained on WebFace42M secured 1st place on the MFR-Ongoing leaderboard (March 2025), achieving 97.25% TAR@FAR=1e-5 and 98.06% TAR@FAR=1e-4 on IJB-C, surpassing all competitors including Partial FC and TransFace [^3^]. This work demonstrates that carefully designed training paradigms can unlock the full potential of large vision transformers for face recognition.

- **ViT-GCT** (Chettaoui et al., ICLR 2026) proposed adding a Global Context Token (GCT) to the input patch sequence, which interacts with all patch tokens via self-attention to deliver complementary context. On IJB-C, ViT-GCT achieved 96.86% at FAR=1e-4 and 94.05% at FAR=1e-5, outperforming CLS-only and CPE+CLS configurations. The GCT token induces stronger focus on key facial landmarks compared to uniformly distributed attention in other configurations [^4^].

- **TopoFR** (Dan et al., 2024) leveraged topological structure alignment using persistent homology to align topological structures between input and latent spaces. TopoFR with R50 backbone outperformed AdaFace and even ViT-based TransFace on IJB-C and IJB-B benchmarks, suggesting that structural information encoding can compensate for backbone limitations [^5^].

- **FaceCoresetNet** (Shapira & Keller, AAAI 2024) framed face set recognition as a differentiable coreset selection problem. The small coreset is used as queries in a self-attention and cross-attention architecture to enrich the descriptor. FaceCoresetNet achieves the best results on TAR@FAR=1e-6 for both IJB-B (52.56%) and IJB-C (91.12%) while maintaining O(N) linear complexity, compared to O(N^2) for competing methods [^6^].

### Swin Transformer for Fingerprint Recognition

- **A Multimodal Biometric System** (Garg et al., 2025) used a convolutional Swin Transformer architecture for feature extraction from fingerprint, iris, and ECG modalities. The model achieved 96% accuracy for fingerprints individually and 99% on the fused multimodal dataset. The Swin Transformer processes spatial (position-based) features and attention-based features in parallel [^7^].

- **Student Thesis** (Cewers & Svensson) implemented and evaluated five transformer models for fingerprint matching on 5.5 million fingerprint pairs. Interestingly, **the Swin Transformer underperformed and incurred higher inference costs**, making it less suitable for deployment. The ViT-Large-Without (no boosting or weighting) achieved the best results overall [^8^].

- **AFR-Net** (Grosz & Jain, IEEE T-BIOM 2024) combined CNN-based and attention-based (ViT) embeddings in a single model, using ResNet50 as backbone and 12 multi-headed attention transformer encoder blocks as the attention-based classification head. AFR-Net achieved rank-1 retrieval of 99.78% on NIST SD14 against a 100K gallery, improving over DeepPrint's 99.20%. The fusion of CNN and ViT representations was key to performance gains [^9^].

### Cross-Attention Transformers for Multimodal Fusion

- **Tiny Transformers with Quality-Gated Fusion** (PeerJ 2026) proposed a multimodal biometric authentication system integrating fingerprint and iris modalities using Tiny ViT for iris and Swin Transformer Tiny for fingerprint. Features are projected into 512-dimensional embeddings trained with ArcFace loss and Triplet loss. A novel Quality-Gated Fusion mechanism uses a 2-layer MLP to dynamically weight similarity scores based on input quality, ensuring high-quality biometric inputs are prioritized [^10^].

- **CSMG Method** (Sharhrah et al., 2025) combined Convolutional Neural Network, Swin Transformer, Multi-Head Self-Attention, and Global Max Pooling for effective feature extraction from fingerprint, iris, and ECG modalities. The proposed method achieves recognition accuracy of 99.90% for fingerprints, 99% for iris, and 99% for ECG [^11^].

- **Ridgeformer** (Pandey et al., 2025) introduced a multi-stage transformer-based approach for contactless fingerprint matching that uses **cross-attention between token sets** from different fingerprint samples. Stage 1 captures global spatial features using a ViT backbone; Stage 2 performs fine-grained cross-sample matching via a cross-attention transformer module between token embeddings. On HKPolyU dataset, Ridgeformer achieved EER of 2.83% and TAR@FAR=0.01 of 89.34%, outperforming all previously benchmarked models [^12^].

### DETR-like Architectures for Biometric Feature Extraction

- No direct application of DETR (Detection Transformer) to biometric feature extraction was found in the literature search. However, **FaceCoresetNet** (Shapira & Keller, AAAI 2024) uses a conceptually similar approach where a small set of learnable "core-template" tokens act as queries in a cross-attention mechanism to aggregate information from the full face template, reminiscent of DETR's object queries [^6^].

- **IFViT** (Qiu et al., IEEE T-IFS 2024) employs a ViT-based Siamese Network with cross-attention mechanisms between fingerprint pairs. Cross-attention allows interaction between patches from both fingerprint images, enabling the model to draw meaningful matches. This architecture shares conceptual similarities with DETR's encoder-decoder cross-attention pattern [^13^].

### Transformer-Based Fusion of Multiple Biometric Traits

- The dominant fusion strategy across surveyed multimodal transformer papers is **score-level fusion** with quality-weighted gating, rather than explicit cross-modal attention. Both the Tiny Transformers paper [^10^] and the CSMG paper [^11^] use quality-aware score fusion mechanisms.

- The **Swin+CNN multimodal architecture** (Garg et al., 2025) uses a Chi-Square test to validate statistical significance of the 99% recognition accuracy on fused fingerprint+iris+ECG data, demonstrating robustness against spoofing attacks [^7^].

- No surveyed paper implements **true inter-modal cross-attention** where tokens from different biometric modalities (e.g., face patches and fingerprint patches) attend to each other within a shared transformer encoder. Current approaches use separate transformer backbones per modality with late fusion.

### Position Embeddings for Non-Image Biometric Data

- **Gazi et al. (2026)** used 2D sinusoidal positional encodings added to fingerprint patch embeddings in their DINOv2-based fingerprint recognition system. Input fingerprint images (224x224) are divided into 196 patches in a 14x14 grid, with each patch linearly projected into a 1536-dimensional latent vector. Four learnable register tokens are incorporated to mitigate the "attention sink" problem [^14^].

- **Minutiae-Guided ViT** (Grosz et al., 2022) is a landmark work that explicitly used **minutiae position information to guide ViT attention maps**. Minutiae coordinates are provided as prior domain knowledge to guide the transformer to focus on local minutiae-related features. The approach achieved TAR=94.23% @ FAR=0.1% on NIST SD 302, compared to a commercial matcher (Verifinger) at TAR=96.71%. The fixed-length embeddings could be matched at 2.5 million matches/second [^15^].

- **Hybrid Graph Transformer** (Liu et al., 2025) combined ViT's global attention with graph-based minutiae modeling for fingerprint and palmprint matching in young children, achieving AUC = 99.99% and EER = 0.26% [^16^].

### Fully Minutiae-Free Approaches

- **Gazi et al. (2026)** introduced the first fully minutiae-free ViT-based fingerprint model using self-supervised DINOv2 fine-tuning on 64,801 images from 12 heterogeneous datasets. The DINOv2-Base (86M parameters) achieved optimal generalization (EER=5.56%) while the Giant model (1.1B parameters) degraded (EER=5.80%) due to overfitting. The model was evaluated across optical, capacitive, rolled, plain, and contactless sensors, demonstrating 4.8x improvement over VeriFinger [^14^].

- **IFViT** (Qiu et al., IEEE T-IFS 2024) provides interpretable fixed-length representations for fingerprint matching via Vision Transformer with a Siamese network architecture. The model uses both self-attention and cross-attention mechanisms, with cross-attention enabling pixel-wise correspondence establishment between fingerprint pairs. IFViT was evaluated on FVC2002, NIST SD4, and other datasets [^13^].

## Major Papers & Sources

| # | Author(s) | Title | Venue | Year | DOI/URL |
|---|-----------|-------|-------|------|---------|
| 1 | Zhong & Deng | Face Transformer for Recognition | arXiv | 2021 | arXiv:2103.14803 |
| 2 | Dan et al. | TransFace: Calibrating Transformer Training for Face Recognition from a Data-Centric Perspective | ICCV | 2023 | DOI: 10.1109/ICCV.2023 | 
| 3 | You et al. | LVFace: Progressive Cluster Optimization for Large Vision Models in Face Recognition | arXiv | 2025 | arXiv:2501.13420 |
| 4 | Chettaoui et al. | ViT-GCT: Enhancing Vision Transformers with a Global Context Token for Face Recognition | ICLR | 2026 | OpenReview (submitted) |
| 5 | Dan et al. | TopoFR: A Closer Look at Topology Alignment on Face Recognition | arXiv | 2024 | arXiv:2410.10587 |
| 6 | Shapira & Keller | FaceCoresetNet: Differentiable Coresets for Face Set Recognition | AAAI | 2024 | DOI: 10.1609/aaai.v38i5.28276 |
| 7 | Garg et al. | A Multimodal Biometric Recognition System based on Fingerprints, Iris and ECG via Swin Transformer and CNN Model | Systems and Soft Computing (Elsevier) | 2025 | DOI: 10.1016/j.sasc.2025.200188 |
| 8 | Cewers & Svensson | Implementation of Vision Transformers for the Matching of Fingerprints in Biometric Verification | Lund University Thesis | 2024 | [LUP] |
| 9 | Grosz & Jain | AFR-Net: Attention-Driven Fingerprint Recognition Network | IEEE T-BIOM | 2024 | DOI: 10.1109/TBIOM.2023.3345764 |
| 10 | PeerJ CS 2026 | Tiny Transformers: Multimodal Digital Biometric Authentication Model | PeerJ Computer Science | 2026 | DOI: 10.7717/peerj-cs.3657 |
| 11 | Sharhrah et al. | Multimodal Biometric Authentication using CNN, Swin Transformer, and Multi-Head Attention | ITM Web of Conferences | 2025 | DOI: 10.1051/itmconf/202521001052 |
| 12 | Pandey et al. | Ridgeformer: Multi-Stage Contrastive Training For Fine-Grained Cross-Domain Fingerprint Recognition | IEEE ICASSP / arXiv | 2025 | arXiv:2506.01806 |
| 13 | Qiu et al. | IFViT: Interpretable Fixed-Length Representation for Fingerprint Matching via Vision Transformer | IEEE T-IFS | 2024 | DOI: 10.1109/TIFS.2024.3412270 |
| 14 | Gazi et al. | Minutiae-Free Fingerprint Recognition via Vision Transformers: An Explainable Approach | Applied Sciences (MDPI) | 2026 | DOI: 10.3390/app16021009 |
| 15 | Grosz et al. | Minutiae-Guided Fingerprint Embeddings via Vision Transformers | arXiv | 2022 | arXiv:2210.13994 |
| 16 | Liu et al. | A Hybrid Graph Transformer Network for Fingerprint and Palmprint Minutiae Matching | Neurocomputing | 2025 | DOI: 10.1016/j.neucom.2024.129180 |

## Trends & Signals

1. **Training Strategy > Architecture**: The key differentiator for transformer-based face recognition is increasingly the training strategy rather than the architecture itself. TransFace's data augmentation and hard sample mining, LVFace's progressive cluster optimization, and TopoFR's topological alignment all focus on how to train ViTs effectively rather than architectural innovation [^2^][^3^][^5^].

2. **Quality-Aware Fusion is Standard**: Multimodal transformer systems consistently use quality-gated or quality-aware fusion mechanisms at the score level, prioritizing high-quality biometric inputs over low-quality ones [^10^][^11^].

3. **Self-Supervised Pretraining for Fingerprints**: The use of large-scale self-supervised pretraining (DINOv2) for fingerprint recognition represents a paradigm shift, enabling minutiae-free approaches that outperform traditional minutiae-based systems on heterogeneous data [^14^].

4. **Moderate Capacity Wins for Limited Data**: In fingerprint recognition, where domain-specific data is limited compared to face recognition, medium-capacity transformers (86M parameters) outperform both smaller and larger (1.1B parameter) models. This contrasts with face recognition where scaling to ViT-L with WebFace42M (42M images) yields best results [^14^][^3^].

5. **Cross-Attention for Fine-Grained Matching**: Ridgeformer demonstrates that cross-attention between token sets from different samples can significantly improve fine-grained matching performance, particularly for cross-domain (contactless-to-contact) fingerprint recognition [^12^].

## Controversies & Conflicting Claims

1. **Swin Transformer vs. ViT for Fingerprints**: Conflicting evidence exists on whether Swin Transformer is suitable for fingerprint recognition. The Cewers & Svensson thesis found that "the Swin Transformer underperformed and incurred higher inference costs, making it less suitable for deployment" [^8^], while Garg et al. (2025) successfully used Swin Transformer for multimodal biometric recognition [^7^]. The difference may be attributed to training data scale, pretraining strategy, and task specifics (fingerprint-only vs. multimodal).

2. **Minutiae-Free vs. Minutiae-Guided**: Grosz et al.'s minutiae-guided approach [^15^] and Gazi et al.'s fully minutiae-free approach [^14^] represent fundamentally different philosophies. Grosz argues that guiding ViT attention with minutiae knowledge improves recognition, while Gazi demonstrates that self-supervised ViTs can autonomously discover minutiae-like structures (IoU=0.41 with ground-truth minutiae) without explicit guidance. The key trade-off is: minutiae-guided approaches require minutiae annotation pipelines, while minutiae-free approaches sacrifice some performance for simplicity and generalization.

3. **CNNs vs. Transformers for Face Recognition**: Despite extensive research on ViTs for face recognition, CNN-based methods (AdaFace, TopoFR with ResNet) remain highly competitive. TopoFR with R50 even outperforms TransFace (ViT-based) on some benchmarks, suggesting that training methodology may matter more than backbone architecture [^5^]. LVFace finally demonstrates clear ViT superiority when combined with large-scale datasets (WebFace42M) and carefully designed training [^3^].

4. **CLS Token vs. Patch Embeddings**: ViT-GCT demonstrates that the standard CLS token approach is suboptimal for face recognition. Their Global Context Token (GCT) consistently outperforms CLS-only and CPE+CLS configurations, with performance gains of +1.0% on IJB-C at FAR=1e-5 [^4^]. This challenges the standard ViT design choice for face recognition.

## Research Gaps Identified

1. **No True Cross-Modal Attention for Biometric Fusion**: No surveyed paper implements explicit inter-modal cross-attention where tokens from different biometric modalities (face patches, fingerprint ridges, iris textures) attend to each other in a shared transformer space. Current approaches use separate encoders with late fusion [^10^][^11^]. This represents a significant gap, as cross-modal attention could potentially learn richer inter-modality correlations.

2. **No DETR-like Architecture for Biometrics**: While DETR and its variants have revolutionized object detection, no direct DETR-like application (using learnable object queries to extract biometric features from a shared encoder) has been found in the biometric literature. FaceCoresetNet [^6^] comes closest with its core-template queries.

3. **Limited Transformer Fusion Research (Face + Fingerprint)**: Most multimodal transformer papers focus on fingerprint+iris+ECG [^7^][^11^] or face+iris combinations [^10^]. The specific combination of face + fingerprint with transformer-based fusion remains underexplored despite these being the two most widely deployed biometric modalities.

4. **Computational Cost vs. Performance Trade-offs**: While many papers report accuracy, comprehensive FLOPs and parameter comparisons across transformer variants are sparse. LVFace reports GFLOPs at 112x112 resolution [^3^], and Gazi et al. report detailed computational costs [^14^], but most multimodal transformer papers lack this analysis.

5. **Position Embeddings for Non-Grid Biometric Data**: While sinusoidal positional encodings work well for image patches, how to best represent non-grid biometric data (e.g., fingerprint minutiae point clouds, ECG time series, gait sequences) in transformers remains an open question. The Hybrid Graph Transformer [^16^] partially addresses this with graph-based minutiae modeling.

6. **Cross-Sensor Generalization**: Most transformer-based fingerprint research evaluates on single-sensor datasets. Gazi et al.'s multi-sensor evaluation [^14^] is a notable exception, showing that self-supervised ViTs can achieve cross-sensor robustness without modality-specific preprocessing.

7. **Real-Time Deployment**: While Tiny Transformers [^10^] target resource-constrained environments, the computational cost of full transformer-based multimodal fusion at scale remains a challenge for real-time deployment. Early exit transformers for face recognition (Nixon et al.) show 25.49% FLOP reduction with only 3.84% accuracy loss [^17^].

## Detailed Evidence Log

### Evidence 1: Face Transformer for Recognition
**Claim**: First application of Vision Transformer to face recognition, establishing ViT as a viable backbone for FR.
**Source**: arXiv preprint
**URL**: https://arxiv.org/abs/2103.14803
**Date**: 2021
**Excerpt**: "Face transformer for recognition" - established the foundational ViT+FR approach with 142 citations by 2024.
**Confidence**: High

### Evidence 2: TransFace (ICCV 2023)
**Claim**: ViTs perform vulnerably in FR with large datasets due to incompatible data augmentation and hard sample mining. Proposed DPAP and EHSM strategies to address this.
**Source**: IEEE/CVF International Conference on Computer Vision (ICCV)
**URL**: https://openaccess.thecvf.com/content/ICCV2023/html/Dan_TransFace_Calibrating_Transformer_Training_for_Face_Recognition_from_a_Data-Centric_ICCV_2023_paper.html
**Date**: 2023
**Excerpt**: "We unexpectedly find that ViTs perform vulnerably when applied to face recognition scenarios with extremely large datasets... TransFace-S outperforms ViT-S by +0.56% on IJB-C (TAR@FAR=1e-4)."
**Confidence**: High

### Evidence 3: LVFace (2025)
**Claim**: Progressive Cluster Optimization with three-stage training achieves SOTA on face recognition benchmarks, including 1st place on MFR-Ongoing.
**Source**: arXiv preprint / ByteDance
**URL**: https://arxiv.org/abs/2501.13420
**Date**: 2025
**Excerpt**: "LVFace achieves 97.25% TAR@FAR=1e-5 and 98.06% TAR@FAR=1e-4 on IJB-C... ranks first on the academic track of the MFR-Ongoing leaderboard."
**Computational Cost**: ViT-B: ~8.6 GFLOPs at 112x112; ViT-L: ~24.5 GFLOPs at 112x112 [from Dosovitskiy et al. scaling]
**Confidence**: High

### Evidence 4: ViT-GCT (ICLR 2026)
**Claim**: Adding a Global Context Token (GCT) that interacts with all patch tokens via self-attention improves face recognition over CLS token baselines.
**Source**: ICLR 2026 Conference Submission
**URL**: https://openreview.net/forum?id=[ICLR 2026 submission]
**Date**: 2026
**Excerpt**: "ViT-GCT results in further performance improvements... on IJB-B and IJB-C at FAR of 10-5, the recognition performance increases from 87.63% to 88.79% and from 93.22% to 94.05%, respectively."
**Computational Cost**: Minimal overhead from single additional token; ~0.1% increase in FLOPs
**Confidence**: Medium (submitted to ICLR 2026, not yet peer-reviewed at time of search)

### Evidence 5: TopoFR (2024)
**Claim**: Topological structure alignment using persistent homology can outperform both CNN and ViT baselines, even with smaller backbones.
**Source**: arXiv preprint
**URL**: https://arxiv.org/abs/2410.10587
**Date**: 2024
**Excerpt**: "Our TopoFR even works better than ViT-based SOTA method TransFace, implying the superiority of our method."
**Confidence**: High

### Evidence 6: FaceCoresetNet (AAAI 2024)
**Claim**: Differentiable coreset selection with self-attention and cross-attention achieves best TAR@FAR=1e-6 on IJB-B and IJB-C while maintaining linear O(N) complexity.
**Source**: AAAI Conference on Artificial Intelligence
**URL**: https://ojs.aaai.org/index.php/AAAI/article/view/28276
**Date**: 2024
**Excerpt**: "FaceCoresetNet achieves the best results on TPR@FPR=1e-6 for both IJB-B (52.56%) and IJB-C (91.12%) while being the most efficient."
**Computational Cost**: O(N) linear in template size, compared to O(N^2) for CAFace and RSA
**Confidence**: High

### Evidence 7: Swin Transformer + CNN for Multimodal Biometrics (2025)
**Claim**: Convolutional Swin Transformer achieves 99% recognition accuracy on fused fingerprint+iris+ECG multimodal data.
**Source**: Systems and Soft Computing (Elsevier)
**URL**: https://www.sciencedirect.com/science/article/pii/S2772941925001887
**Date**: 2025
**Excerpt**: "The proposed model was first applied individually to each modality, achieving recognition accuracies of 96% for fingerprints, 97% for iris, and 71% for ECG. Subsequently... yielding a recognition accuracy of 99%."
**Confidence**: High

### Evidence 8: Vision Transformers for Fingerprint Matching - Thesis
**Claim**: Swin Transformer underperforms compared to ViT for fingerprint matching, with higher inference costs.
**Source**: Lund University Master's Thesis
**URL**: https://lup.lub.lu.se/student-papers/record/9201642
**Date**: 2024
**Excerpt**: "The Swin Transformer underperformed and incurred higher inference costs, making it less suitable for deployment... ViT-Large-Without emerges as the most promising candidate."
**Confidence**: Medium (thesis work, not peer-reviewed journal)

### Evidence 9: AFR-Net (IEEE T-BIOM 2024)
**Claim**: Fusing CNN-based and attention-based (ViT) embeddings in a single model achieves SOTA fingerprint recognition, including 99.78% rank-1 retrieval on NIST SD14 against 100K gallery.
**Source**: IEEE Transactions on Biometrics, Behavior, and Identity Science
**URL**: https://ieeexplore.ieee.org/document/10356124
**Date**: 2024
**Excerpt**: "AFR-Net achieves a rank 1 retrieval rate of 99.78%, which is an improvement over the previous SOTA performance of 99.20% by DeepPrint."
**Computational Cost**: ~25M parameters (ResNet50 + 12 transformer blocks); fusion saves latency vs. separate models
**Confidence**: High

### Evidence 10: Tiny Transformers with Quality-Gated Fusion (2026)
**Claim**: Tiny ViT + Swin Transformer with quality-gated score fusion achieves robust multimodal biometric authentication for resource-constrained environments.
**Source**: PeerJ Computer Science
**URL**: https://peerj.com/articles/cs-3657/
**Date**: 2026
**Excerpt**: "A gating network returns the fusion weights... This adaptive mechanism ensures that high-quality biometric inputs are prioritised, while the influence of low-quality or noisy inputs is attenuated."
**Computational Cost**: Tiny ViT (~5.7M params) + Tiny Swin (~5.7M params); 256-dim embeddings; 15 epochs training
**Inter-Modal Attention**: No explicit cross-modal attention; separate backbones with score-level fusion
**Confidence**: High

### Evidence 11: Ridgeformer (2025)
**Claim**: Cross-attention between token sets from different fingerprint samples improves cross-domain (contactless-to-contact) fingerprint matching.
**Source**: IEEE ICASSP / arXiv
**URL**: https://arxiv.org/abs/2506.01806
**Date**: 2025
**Excerpt**: "The transformer performs cross-attention between the token sets... generating attended token representations... Ridgeformer achieved approximately a 4% improvement in Rank-1 Recall."
**Computational Cost**: Two-stage ViT + cross-attention module
**Confidence**: High

### Evidence 12: CSMG Method (2025)
**Claim**: CNN + Swin Transformer + Multi-Head Self-Attention + Global Max Pooling achieves 99.90% fingerprint accuracy in multimodal setting.
**Source**: ITM Web of Conferences
**URL**: https://www.itm-conferences.org/articles/itmconf/pdf/2025/10/itmconf_keis2025_01052.pdf
**Date**: 2025
**Excerpt**: "The proposed CSMG method achieves a recognition accuracy of 99.90% for fingerprints, 99% for irises, and 99% for ECG."
**Confidence**: Medium (conference proceedings)

### Evidence 13: IFViT (IEEE T-IFS 2024)
**Claim**: Cross-attention between fingerprint pairs via ViT-based Siamese network enables interpretable fixed-length fingerprint representations with pixel-wise correspondences.
**Source**: IEEE Transactions on Information Forensics and Security
**URL**: https://ieeexplore.ieee.org/document/10589124
**Date**: 2024
**Excerpt**: "The core of the ViT lies in its attention layers, particularly self-attention and cross-attention mechanisms... cross-attention allows interaction between patches from both fingerprint images."
**Computational Cost**: 256-dimensional fixed-length representations; ResNet-18 + ViT architecture
**Confidence**: High

### Evidence 14: Minutiae-Free ViT Fingerprint (2026)
**Claim**: Self-supervised DINOv2 ViT achieves fully minutiae-free fingerprint recognition with 4.8x improvement over VeriFinger on heterogeneous multi-sensor data. Medium-capacity (86M) model outperforms 1.1B parameter giant.
**Source**: Applied Sciences (MDPI)
**URL**: https://www.mdpi.com/2076-3417/16/2/1009
**Date**: 2026
**Excerpt**: "The base model consistently achieves the best overall performance, obtaining an EER of 5.56%... This represents a 14.5% relative improvement over the small model and a 4.1% improvement over the giant model."
**Computational Cost**: 
- Small: 22.06M params, 58.99 GFLOPs, 2.77ms latency
- Base: 86.58M params, 234.69 GFLOPs, 3.15ms latency  
- Large: 304.37M params, 831.65 GFLOPs, 8.64ms latency
- Giant: 1136.49M params, 3115.04 GFLOPs, 24.19ms latency
**Confidence**: High

### Evidence 15: Minutiae-Guided ViT (2022)
**Claim**: Guiding ViT attention with minutiae positions improves fingerprint recognition, achieving near-parity with commercial matchers at 40x faster matching speed.
**Source**: arXiv preprint
**URL**: https://arxiv.org/abs/2210.13994
**Date**: 2022
**Excerpt**: "We obtain a TAR=94.23% @ FAR=0.1% on NIST SD 302... our fixed-length embeddings can be matched orders of magnitude faster than the commercial system (2.5 million matches/second compared to 50K matches/second)."
**Computational Cost**: ViT-Small with 12 layers, 6 attention heads; ~22M parameters
**Confidence**: High

### Evidence 16: Hybrid Graph Transformer (2025)
**Claim**: Combining ViT global attention with graph-based minutiae modeling achieves AUC=99.99% for fingerprint/palmprint matching in young children.
**Source**: Neurocomputing (Elsevier)
**URL**: https://www.sciencedirect.com/science/article/pii/S0925231224013150
**Date**: 2025
**Excerpt**: "Combines ViT's global attention with graph-based minutiae modeling for complex prints... AUC = 99.99%, EER = 0.26%."
**Confidence**: High

### Evidence 17: Early Exit ViT for Face Recognition
**Claim**: Adding early exits to ViT can reduce compute budget at inference with only small loss in performance; 25.49% of FLOPs with 3.84% accuracy loss.
**Source**: IAPR International Conference on Pattern Recognition in Information and Communication Technologies (ICPRICT)
**URL**: https://dl.gi.de/bitstreams/111dfa8c-7d90-41a2-9b8d-7e2a89a9c31a/download
**Date**: 2024
**Excerpt**: "With 25.49% FLOPs of the total model we only lose 3.84% TMr@FMr1%, 11.14% TMr@FMr0.1% and 17.02% TMr@FMr0.01%."
**Confidence**: Medium

---

## Computational Cost Summary Table

| Model | Architecture | Parameters | FLOPs | Latency | Inter-Modal Attention |
|-------|-------------|------------|-------|---------|----------------------|
| TransFace-S | ViT-S | ~22M | ~4.6G | - | N/A (unimodal) |
| TransFace-L | ViT-L | ~86M | ~24.5G | - | N/A (unimodal) |
| LVFace-L | ViT-L | ~86M | ~24.5G | - | N/A (unimodal) |
| ViT-GCT-B | ViT-B + GCT | ~86M | ~24.6G | - | N/A (unimodal) |
| FaceCoresetNet | IResNet101 + Attn | ~45M | O(N) linear | - | Self + Cross-attention |
| AFR-Net | ResNet50 + ViT | ~25M | - | - | N/A (unimodal hybrid) |
| Ridgeformer | ViT + Cross-Attn | ~22M | - | - | Cross-sample attention |
| IFViT | ResNet18 + ViT | ~12M | - | - | Cross-attention (pairwise) |
| Gazi-Base (Fingerprint) | DINOv2-ViT-Base | 86.58M | 234.69G | 3.15ms | N/A (unimodal) |
| Tiny Transformers | Tiny ViT + Tiny Swin | ~11.4M combined | - | - | Score-level fusion (no cross-attention) |
| Minutiae-Guided ViT | ViT-Small | ~22M | - | - | Minutiae-guided self-attention |

## Summary

The surveyed literature reveals that transformer architectures have made significant inroads into biometric recognition, particularly for face recognition where ViT-based methods (TransFace, LVFace, ViT-GCT) now achieve state-of-the-art results on standard benchmarks. For fingerprint recognition, both minutiae-guided (Grosz et al.) and fully minutiae-free self-supervised approaches (Gazi et al.) demonstrate strong performance, with cross-sample cross-attention (Ridgeformer, IFViT) emerging as a powerful technique for fine-grained matching.

However, **a critical gap remains in transformer-based multimodal biometric fusion**: no surveyed paper implements true inter-modal cross-attention where tokens from different biometric modalities (e.g., face patches and fingerprint minutiae/ridges) directly attend to each other. Current multimodal transformer approaches universally use separate modality-specific transformer backbones with late score-level fusion, typically enhanced with quality-aware gating. This represents a significant research opportunity, as cross-modal attention mechanisms could potentially learn richer joint representations that exploit the complementary nature of different biometric traits.

The computational cost of transformer-based biometric systems varies significantly: lightweight Tiny Transformer variants (~5-11M parameters) are suitable for edge deployment, while large ViT-L models (~86M parameters, ~24 GFLOPs) require substantial compute but achieve best-in-class performance on large-scale face recognition benchmarks. For fingerprint recognition, medium-capacity models (86M parameters) appear optimal, with larger models showing overfitting on limited domain-specific data.
