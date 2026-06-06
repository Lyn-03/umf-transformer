## Dimension 6: Robustness to Degraded Face Data in Biometric Systems

### Scope
This dimension examines approaches for handling degraded, occluded, or low-quality face images in biometric verification systems (2020-2026). The focus is on methods that either improve recognition under degradation or provide quality estimates that can modulate multimodal fusion weights, particularly for face+fingerprint fusion.

---

### Key Findings

**1. Quality-Adaptive Loss Functions Show Strong Promise for Degraded Face Recognition**
- AdaFace (CVPR 2022) [^174^] proposes a quality-adaptive margin that adjusts emphasis on hard samples based on image quality approximated via feature norms. It achieves SoTA on IJB-B, IJB-C, IJB-S, and TinyFace, improving low-quality dataset performance while maintaining high-quality performance. The feature norm serves as an implicit quality estimator that could modulate fusion weights.
- MagFace (CVPR 2021) [^177^] introduces a magnitude-aware angular margin loss that generates quality-aware embeddings. QMagFace (WACV 2023) [^76^] extends this with a quality-aware comparison score, achieving 98.74% on CFP-FP and 83.95% on XQLFW, demonstrating especially strong performance under cross-quality matching scenarios.
- ElasticFace (CVPR 2022) [^229^] relaxes fixed-margin constraints by sampling margins from a Gaussian distribution, improving generalization across seven of nine benchmarks. ElasticFace is more robust to inconsistent intra/inter-class variations common in degraded data.

**2. Face Image Quality Assessment (FIQA) Has Matured Significantly**
- SER-FIQ (CVPR 2020) [^175^] pioneered unsupervised FIQA using stochastic embedding robustness from random subnetworks of a face model. It outperforms supervised baselines and avoids quality label requirements entirely, making it highly practical for deployment.
- CR-FIQA (CVPR 2023) [^199^] learns sample relative classifiability by probing internal network observations during training. It ranked 1st/2nd at NIST FATE Quality evaluation, and predicts quality by estimating how close a sample would be to its class center. It is the current SoTA for model-specific FIQA.
- SDD-FIQA (CVPR 2021) [^201^] uses Wasserstein distance between intra-class and inter-class similarity distributions as quality pseudo-labels. It outperforms earlier methods by considering both class information sources simultaneously.
- GraFIQs (CVPR 2024W) [^233^] introduces a completely training-free approach using gradient magnitudes from Batch Normalization statistics discrepancy. It achieves competitive results with SoTA without any quality labeling or regression network training.

**3. Occlusion-Robust Face Recognition Has Advanced Through Dataset and Method Innovation**
- WebFace-OCC (ICASSP 2021) [^200^] is a simulated occlusion dataset with 804,704 images of 10,575 subjects, combining diverse occlusion types (masks, glasses, textures, colors). ArcFace retrained on WebFace-OCC significantly outperforms SoTA on occluded face benchmarks.
- ALOB (Sensors 2023) [^241^] proposes adversarial learning of occlusions via backpropagation without manual occlusion labeling. It achieves 100% recognition on AR sunglasses protocols and 98.77% on LFW with occlusions.
- Partial face recognition approaches (e.g., Liao & Jain's MKD-SRC framework) [^236^] adopt alignment-free strategies using keypoint descriptors, enabling recognition of arbitrary face patches without requiring full face alignment.

**4. Low-Resolution and Surveillance Face Recognition Require Specialized Approaches**
- SlackedFace (BMVC 2023) [^244^] introduces a "slacked margin" for low-resolution face recognition that de-emphasizes unrecognizable hard examples while emphasizing recognizable ones. It achieves 80% rank-1 on extended SCFace D1 (vs. 78.75% for AdaFace) and provides a reliable recognizability index (P-Norm).
- Recognizability Embedding Enhancement (CVPR 2023) [^271^] formulates a recognizability index (RI) based on proximity to unrecognizable face clusters and class prototypes, achieving SoTA on three challenging low-resolution datasets.
- IJB-S [^273^] provides a surveillance video benchmark with 350 videos spanning 30 hours, enabling evaluation under realistic low-quality surveillance conditions.

**5. Synthetic Degradation and Data Augmentation Are Critical for Robust Training**
- XQLFW (FG 2021) [^267^] demonstrates that synthetic degradation using realistic Gaussian anisotropic kernels for downsampling better reflects real-world low quality than simple bicubic downsampling. Performance on XQLFW drops 25.28% for ArcFace compared to LFW, highlighting the severity of quality degradation effects.
- Face restoration methods (GFP-GAN, DifFace, VQFR, CodeFormer) [^224^] can serve as preprocessing pipelines. DifFace (diffusion-based) achieves the best FID scores but requires 11.38s inference; GFP-GAN provides a practical speed/quality tradeoff at 0.06s.

**6. Quality Scores Can Directly Modulate Fusion Weights**
- A Quality-Guided Mixture of Score-Fusion Experts framework (2025) [^144^] demonstrates that quality weights (e.g., AdaFace quality weights) can effectively guide score-fusion in multimodal human recognition, prioritizing more reliable modalities based on quality estimates.
- MagFace's magnitude-aware embeddings naturally encode quality information suitable for fusion weighting [^177^].
- The NIST FATE Quality evaluation [^226^] formally validates that discarding low-quality face images (1-5%) reduces FNMR by 0.2-0.4%, confirming the operational value of quality-aware processing.

---

### Major Papers & Sources

1. **Kim, M., Jain, A.K., Liu, X.**, "AdaFace: Quality Adaptive Margin for Face Recognition," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. URL: https://arxiv.org/abs/2204.00964

2. **Terhörst, P., Kolf, J.N., Damer, N., Kirchbuchner, F., Kuijper, A.**, "SER-FIQ: Unsupervised Estimation of Face Image Quality Based on Stochastic Embedding Robustness," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 5650-5659, 2020. DOI: 10.1109/CVPR42600.2020.00569

3. **Boutros, F., Fang, M., Klemt, M., Fu, B., Damer, N.**, "CR-FIQA: Face Image Quality Assessment by Learning Sample Relative Classifiability," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023. URL: https://arxiv.org/abs/2112.06592

4. **Ou, F.-Z., Chen, X.**, "SDD-FIQA: Unsupervised Face Image Quality Assessment with Similarity Distribution Distance," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021. URL: https://arxiv.org/abs/2103.05977

5. **Kolf, J.N., Damer, N., Boutros, F.**, "GraFIQs: Face Image Quality Assessment Using Gradient Magnitudes," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, pp. 1490-1499, 2024. DOI: 10.1109/CVPRW63382.2024.00156

6. **Terhörst, P.**, "Simple and Accurate Quality-Aware Face Recognition (QMagFace)," *arXiv:2111.13475*, 2021; also presented at WACV 2023 as "QMagFace: Simple and Accurate Quality-Aware Face Recognition." URL: https://arxiv.org/abs/2111.13475

7. **Boutros, F.**, "ElasticFace: Elastic Margin Loss for Deep Face Recognition," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022. URL: https://arxiv.org/abs/2109.09416

8. **Huang, B., Wang, Z., Wang, G., Jiang, K., Zeng, K., Han, Z., Tian, X., Yang, Y.**, "When Face Recognition Meets Occlusion: A New Benchmark," *Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*, pp. 4240-4244, 2021. URL: https://arxiv.org/abs/2103.02805

9. **Zhao, C., Qin, Y., Zhang, B.**, "Adversarially Learning Occlusions by Backpropagation for Face Recognition," *Sensors*, vol. 23, no. 20, 8559, 2023. DOI: 10.3390/s23208559

10. **Low, C.Y., Chai, J.C.L., Park, J., An, K., Cha, M.**, "SlackedFace: Learning a Slacked Margin for Low-Resolution Face Recognition," *Proc. British Machine Vision Conference (BMVC)*, 2023. URL: https://papers.bmvc2023.org/0282.pdf

11. **Chai, J.C.L., Ng, T.S., Low, C.Y., Park, J., Teoh, A.B.J.**, "Recognizability Embedding Enhancement for Very Low-Resolution Face Recognition and Quality Estimation," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 9957-9967, 2023.

12. **Knoche, M., Hörmann, S., Rigoll, G.**, "Cross-Quality LFW: A Database for Analyzing Cross-Resolution Image Face Recognition in Unconstrained Environments," *Proc. IEEE International Conference on Automatic Face and Gesture Recognition (FG)*, 2021. DOI: 10.1109/FG52635.2021.9666960

13. **Kalka, N.D., Maze, B., Duncan, J.A., O'Connor, K., Elliott, S., Hebert, K., Bryan, J., Jain, A.K.**, "IJB-S: IARPA Janus Surveillance Video Benchmark," *Proc. IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS)*, 2018. DOI: 10.1109/BTAS.2018.8698584

14. **Huang, Y., Wang, Z., Tai, Y., Liu, X., Shen, P., Li, S., Li, J., Huang, F.**, "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

15. **Meng, Q., Zhao, S., Huang, Z., Zhou, F.**, "MagFace: A Universal Representation for Face Recognition and Quality Assessment," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021.

16. **Cheng, Y., Zhu, S., Gong, X.**, "TinyFace: Benchmarking Face Recognition on Low-Resolution Images," *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 14235-14244, 2021. [CITATION TO VERIFY]

17. **Schlett, T., Rathgeb, C., Damer, N.**, "Evaluating Face Image Quality Score Fusion for Modern Face Recognition Systems," *Proc. International Conference on Pattern Recognition (ICPR)*, 2021. URL: https://dl.gi.de/bitstreams/ce99654c-d3cf-499e-9d35-71420f5e9b4e/download

18. **Wang, X., et al.**, "A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition," *arXiv:2508.00053*, 2025. URL: https://arxiv.org/html/2508.00053v1

19. **Yasarla, R., Perazzi, F., Patel, V.M.**, "Deblurring Face Images Using Uncertainty Guided Multi-Stream Semantic Networks," *IEEE Transactions on Image Processing*, vol. 29, pp. 6251-6263, 2020.

20. **Liao, S., Jain, A.K.**, "Partial Face Recognition: An Alignment Free Approach," *IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)*, vol. 35, no. 5, pp. 1193-1205, 2013. DOI: 10.1109/TPAMI.2012.191

---

### Trends & Signals

**Trend 1: Shift from Supervised to Unsupervised/Learning-Free FIQA**
The field has rapidly moved from supervised FIQA methods requiring human-labeled quality scores (error-prone) to unsupervised approaches (SER-FIQ, SDD-FIQA) and completely training-free approaches (GraFIQs). This trend is driven by the realization that quality labels are inherently ambiguous [^175^]. GraFIQs represents the extreme end of this spectrum, requiring no training whatsoever while achieving competitive performance [^233^].

**Trend 2: Integration of Quality Awareness into Loss Functions**
Rather than treating quality estimation as a separate module, recent approaches embed quality awareness directly into the training loss: AdaFace adjusts margins by feature norm [^174^], MagFace uses embedding magnitude to encode quality [^177^], SlackedFace introduces recognizability-aware slacked margins [^244^], and CurricularFace implements adaptive curriculum learning [^232^]. This integration eliminates the need for separate quality networks.

**Trend 3: Synthetic Degradation for Robust Training**
The emergence of large-scale synthetically degraded datasets (WebFace-OCC [^200^], XQLFW [^267^]) has enabled systematic training for degradation robustness. Synthetic occlusion using diverse object types, textures, and colors has proven more effective than simple downsampling for improving real-world robustness [^209^].

**Trend 4: Face Restoration as Preprocessing for Recognition**
Face restoration methods (GFP-GAN, DifFace, VQFR, CodeFormer) are increasingly used as preprocessing pipelines before recognition [^224^]. Diffusion-based methods (DifFace) achieve the highest perceptual quality but at significant computational cost; GAN-based methods (GFP-GAN) offer better speed/quality tradeoffs. Identity preservation during restoration remains a key challenge.

**Trend 5: Quality Scores as Fusion Modulators**
Recent work explicitly demonstrates that FIQA scores can guide multimodal fusion weights [^144^]. Quality-guided mixture-of-experts frameworks show that predicted quality weights from AdaFace effectively prioritize more reliable modalities. This directly supports the use of face quality scores for face+fingerprint fusion weight modulation.

---

### Controversies & Conflicting Claims

**Controversy 1: What Constitutes "Quality" for Face Recognition?**
There is fundamental disagreement about how to define face image quality. Early approaches used human-interpretable factors (blur, illumination, pose) [^151^], while recent "monolithic" FIQA methods (SER-FIQ, CR-FIQA, MagFace) estimate holistic utility for a specific recognition model [^151^]. SER-FIQ demonstrated that model-specific quality assessment significantly outperforms generic quality metrics [^175^], suggesting that quality is inherently coupled to the recognition system rather than being an intrinsic image property.

**Controversy 2: Feature Norm as Quality Proxy -- Is It Sufficient?**
AdaFace [^174^] and MagFace [^177^] use feature norm (embedding magnitude) as a proxy for image quality. However, SlackedFace [^244^] argues that embedding norm alone is insufficient: unrecognizable hard examples may have large norms, and recognizable hard examples may have small norms. SlackedFace proposes combining norm with embedding proximity (P-Norm) for better quality estimation. This conflict suggests that simple norm-based quality may fail in extreme degradation scenarios.

**Controversy 3: Synthetic vs. Real Degradation for Training**
The effectiveness of synthetic degradation for training remains debated. XQLFW [^267^] argues that simple bicubic downsampling is unrealistic and proposes Gaussian anisotropic kernels. However, real-world surveillance data (IJB-S [^273^]) contains complex degradation patterns (motion blur, compression, low light) that synthetic methods may not capture. FanVID [^243^] (2025) proposes video-based benchmarks where individual frames are genuinely unrecognizable, requiring temporal integration.

**Controversy 4: Restoration vs. Robust Recognition -- Which Approach?**
Two competing paradigms exist: (a) restore degraded faces before recognition using super-resolution/deblurring [^247^], or (b) train recognition models to be inherently robust to degradation [^174^][^244^]. Restoration risks identity drift (hallucinated features), while robust recognition may fail under extreme degradation. Current consensus favors robust recognition with optional restoration preprocessing [^224^].

**Controversy 5: Fixed vs. Adaptive Margins**
Traditional approaches (ArcFace, CosFace) use fixed margins, while newer methods (ElasticFace [^229^], AdaFace [^174^], SlackedFace [^244^]) advocate adaptive margins. X2-Softmax [^227^] argues that Gaussian-sampled margins in ElasticFace may be redundant; the debate centers on whether margin randomness genuinely improves generalization or simply adds noise.

---

### Research Gaps Identified

**Gap 1: Cross-Modality Quality Correlation for Fusion**
While face FIQA has matured, there is limited work on how face quality scores correlate with fingerprint quality in multimodal fusion scenarios. Most quality-aware fusion frameworks assume independent quality estimation per modality [^144^]. Research is needed on joint quality estimation that considers cross-modality correlations.

**Gap 2: Quality Estimation for Extreme Degradation**
Current FIQA methods struggle to reliably distinguish between "very low quality" and "extremely low quality" images, both of which are effectively unrecognizable. SlackedFace's P-Norm [^244^] makes progress here, but quality estimation in the surveillance-to-extreme range remains unreliable.

**Gap 3: Temporal Quality Aggregation for Video**
For surveillance video (IJB-S [^273^]), most approaches aggregate frame-level quality scores (e.g., averaging). Optimal temporal quality aggregation strategies that exploit temporal redundancy for quality estimation remain underexplored.

**Gap 4: Real-Time Quality-Aware Fusion**
While CR-FIQA achieves high accuracy, it requires training a quality regression network. GraFIQs [^233^] is training-free but still requires backpropagation through the FR model. Lightweight quality estimators suitable for real-time multimodal fusion deployment are needed.

**Gap 5: Occlusion + Degradation Compound Effects**
Most work treats occlusion and quality degradation (blur, noise, low resolution) separately. The compound effect of simultaneous occlusion AND low quality (e.g., masked faces in surveillance video) is underexplored. WebFace-OCC [^200^] addresses occlusion but at high quality; IJB-S [^273^] addresses low quality but with limited occlusion.

**Gap 6: Standardized Quality-Fusion Benchmarks**
No standardized benchmark exists for evaluating quality-aware multimodal biometric fusion. NIST FATE evaluates FIQA in isolation [^226^]; the community needs benchmarks that explicitly test quality-guided fusion for face+fingerprint systems.

---

### Detailed Evidence Log

#### Entry 1: AdaFace
Claim: A loss function that assigns different importance to samples of different difficulties based on their image quality, approximating quality via feature norms, improves recognition on low-quality datasets while maintaining high-quality performance.
Source: CVPR 2022
URL: https://arxiv.org/abs/2204.00964
Date: 2022
Excerpt: "We propose a new loss function that emphasizes samples of different difficulties based on their image quality. Our method achieves this in the form of an adaptive margin function by approximating the image quality with feature norms. Extensive experiments show that our method, AdaFace, improves the face recognition performance over the state-of-the-art on four datasets (IJB-B, IJB-C, IJB-S and TinyFace)."
Confidence: High

#### Entry 2: SER-FIQ
Claim: Unsupervised face image quality assessment using stochastic embedding robustness from random subnetworks outperforms supervised approaches and avoids quality label requirements.
Source: CVPR 2020
URL: https://arxiv.org/abs/2003.09373
Date: 2020
DOI: 10.1109/CVPR42600.2020.00569
Excerpt: "Avoiding the use of inaccurate quality labels, we proposed a novel concept to measure face quality based on an arbitrary face recognition model. By determining the embedding variations generated from random subnetworks of a face model, the robustness of a sample representation and thus, its quality is estimated."
Confidence: High

#### Entry 3: CR-FIQA
Claim: Learning to predict sample relative classifiability (proximity to class center vs. nearest negative class center) during training produces superior FIQA that ranked 1st/2nd at NIST FATE Quality evaluation.
Source: CVPR 2023
URL: https://arxiv.org/abs/2112.06592
Date: 2023
Excerpt: "We propose a novel learning paradigm that learns internal network observations during the training process. Our proposed CR-FIQA uses this paradigm to estimate the face image quality of a sample by predicting its relative classifiability."
Confidence: High

#### Entry 4: SDD-FIQA
Claim: Wasserstein distance between intra-class and inter-class similarity distributions provides effective quality pseudo-labels for unsupervised FIQA.
Source: CVPR 2021
URL: https://arxiv.org/abs/2103.05977
Date: 2021
Excerpt: "We propose a novel unsupervised FIQA method that incorporates Similarity Distribution Distance for Face Image Quality Assessment (SDD-FIQA). Our method generates quality pseudo-labels by calculating the Wasserstein Distance between the intra-class similarity distributions and inter-class similarity distributions."
Confidence: High

#### Entry 5: GraFIQs
Claim: A completely training-free FIQA approach using gradient magnitudes from Batch Normalization statistics discrepancy achieves competitive results with SoTA.
Source: CVPR 2024 Workshop on Biometrics
URL: https://arxiv.org/abs/2404.12203
Date: 2024
DOI: 10.1109/CVPRW63382.2024.00156
Excerpt: "We propose quantifying the discrepancy in Batch Normalization statistics (BNS) between those recorded during FR training and those obtained from testing samples. We then generate gradient magnitudes of pretrained FR weights by backpropagating the BNS through the pretrained model. The cumulative absolute sum serves as the FIQ."
Confidence: High

#### Entry 6: QMagFace / MagFace
Claim: Combining quality-aware comparison scores with magnitude-aware angular margin loss produces SoTA results especially under cross-quality matching (98.74% CFP-FP, 83.95% XQLFW).
Source: WACV 2023 (QMagFace); CVPR 2021 (MagFace)
URL: https://arxiv.org/abs/2111.13475
Date: 2021/2023
Excerpt: "The proposed approach includes model-specific face image qualities in the comparison process to enhance the recognition performance under unconstrained circumstances. Exploiting the linearity between the qualities and their comparison scores induced by the utilized loss, our quality-aware comparison function is simple and highly generalizable."
Confidence: High

#### Entry 7: ElasticFace
Claim: Elastic margins drawn from a Gaussian distribution improve face recognition robustness over fixed margins by allowing flexible class separability learning.
Source: CVPR 2022
URL: https://arxiv.org/abs/2109.09416
Date: 2022
Excerpt: "We relax the fixed penalty margin constrain by proposing elastic penalty margin loss (ElasticFace) that allows flexibility in the push for class separability. The main idea is to utilize random margin values drawn from a normal distribution in each training iteration."
Confidence: High

#### Entry 8: WebFace-OCC
Claim: Training on a large-scale synthetically occluded dataset (804,704 images, 10,575 subjects) significantly improves face recognition under occlusion.
Source: ICASSP 2021
URL: https://arxiv.org/abs/2103.02805
Date: 2021
Excerpt: "We pioneer a simulated occlusion face recognition dataset... Webface-OCC covers 804,704 face images of 10,575 subjects, with diverse occlusion types to ensure its diversity and stability. Extensive experiments show that the ArcFace retrained by our dataset significantly outperforms the state-of-the-arts."
Confidence: High

#### Entry 9: ALOB
Claim: Adversarial learning of occlusions via backpropagation without manual labeling achieves competitive results for occluded face recognition.
Source: Sensors, 2023
URL: https://www.mdpi.com/1424-8220/23/20/8559
Date: 2023
DOI: 10.3390/s23208559
Excerpt: "We propose an Adversarially Learning Occlusions by Backpropagation (ALOB) model... it learns the corrupted features against personal identity labels, thereby maximizing the loss. For the AR datasets, the ALOB model outperformed other advanced methods by obtaining a 100% recognition rate for images with sunglasses."
Confidence: High

#### Entry 10: SlackedFace
Claim: A slacked margin that de-emphasizes unrecognizable hard examples based on combined embedding norm and proximity (P-Norm) improves low-resolution face recognition.
Source: BMVC 2023
URL: https://papers.bmvc2023.org/0282.pdf
Date: 2023
Excerpt: "We propose SlackedFace to induce a relaxed margin aligned with face recognizability and the model's confidence based on both embedding norm and embedding proximity. SlackedFace achieves 80.00% R1 on extended SCFace D1 compared to 78.75% for AdaFace."
Confidence: High

#### Entry 11: Recognizability Embedding Enhancement
Claim: Formulating a recognizability index based on proximity to unrecognizable clusters and class prototypes achieves SoTA on very low-resolution face recognition.
Source: CVPR 2023
URL: https://openaccess.thecvf.com/content/CVPR2023/html/Chai_2023_CVPR_paper.html
Date: 2023
Excerpt: "We formulate a robust learning-based face recognizability measure, namely recognizability index (RI), based on two criteria: (i) proximity of each face embedding against the unrecognizable faces cluster center and (ii) closeness of each face embedding against its positive and negative class prototypes."
Confidence: High

#### Entry 12: XQLFW
Claim: Cross-quality face recognition evaluation reveals that performance on standard LFW does not predict cross-quality performance; ArcFace drops 25.28% on XQLFW.
Source: FG 2021
URL: https://arxiv.org/pdf/2108.10290
Date: 2021
DOI: 10.1109/FG52635.2021.9666960
Excerpt: "With XQLFW, we show that these models perform differently in cross-quality cases, and hence, the generalizing capability is not accurately predicted by their performance on LFW."
Confidence: High

#### Entry 13: IJB-S
Claim: Surveillance video face recognition requires handling extremely low-quality frames; even state-of-the-art detectors and recognizers achieve only 4-32% rank-5 accuracy on surveillance-to-single protocols.
Source: BTAS 2018
URL: https://arxiv.org/abs/1812.04058
Date: 2018
DOI: 10.1109/BTAS.2018.8698584
Excerpt: "The MTCCN and FaceNet algorithm combination fails to provide high retrieval rates through the first 50 ranks. Rank 5 performance increases to 20% with ground truth detections, and up to 32% with Speco-only partitioning and ground truth bounding boxes."
Confidence: High

#### Entry 14: CurricularFace
Claim: Adaptive curriculum learning that automatically categorizes easy/hard samples and adjusts their learning importance improves face recognition.
Source: CVPR 2020
URL: http://cvlab.cse.msu.edu/pdfs/huang_wang_tai_liu_shen_li_li_huang_cvpr2020.pdf
Date: 2020
Excerpt: "CurricularFace adopts the idea of curriculum learning and casts the well-known negative cosine in face recognition into an adaptive curriculum function. Easy samples are first learned, and hard samples are progressively emphasized."
Confidence: High

#### Entry 15: Quality-Guided Fusion Framework
Claim: Quality weights from FIQA methods (e.g., AdaFace quality weights) can effectively guide multimodal score-fusion, prioritizing more reliable modalities based on quality estimates.
Source: arXiv 2025
URL: https://arxiv.org/html/2508.00053v1
Date: 2025
Excerpt: "The use of ranking-based pseudo-labels encourages the model to focus on relative quality, making it more robust to outliers. This guides the score-fusion experts to prioritize the most reliable modality based on quality."
Confidence: Medium (recent preprint)

#### Entry 16: FIQA Score Fusion Evaluation
Claim: Fusing multiple FIQA method outputs produces more reliable quality scores than any single method.
Source: ICPR 2021
URL: https://dl.gi.de/bitstreams/ce99654c-d3cf-499e-9d35-71420f5e9b4e/download
Date: 2021
Excerpt: "Quality score fusion takes a number of quality scores as input and generates a 'fused' quality score as output. Each FIQA method fusion configuration can thus itself be considered as a new FIQA method."
Confidence: High

#### Entry 17: Face Restoration Survey
Claim: Diffusion-based face restoration (DifFace) achieves the best perceptual quality but requires 11.38s per image; GFP-GAN provides the best practical speed/quality tradeoff at 0.06s.
Source: arXiv 2025 (comprehensive survey)
URL: https://arxiv.org/html/2211.02831v3
Date: 2025
Excerpt: "DifFace (diffusion-based) achieves the highest performance in terms of FID and SSIM... Diffusion-based methods can yield competitive perceptual quality and improved distribution alignment, but typically require iterative sampling at inference."
Confidence: High

#### Entry 18: NIST FATE Quality
Claim: Discarding 1-5% of lowest quality face images reduces FNMR by 0.2-0.4% across 15 face verification algorithms.
Source: NIST FATE Quality Evaluation (ongoing)
URL: https://pages.nist.gov/frvt/html/frvt_quality.html
Date: 2023
Excerpt: "The testing scenario examined the impact of eliminating 1% and 5% of the poorest quality facial images, as determined by a QAA, on the FNMR of 15 facial recognition algorithms."
Confidence: High

#### Entry 19: Partial Face Recognition
Claim: Alignment-free partial face recognition using multi-keypoint descriptors achieves 81.31% rank-1 accuracy on FRGCv2.0 partial faces without requiring eye alignment.
Source: IEEE T-PAMI, 2013
URL: https://biometrics.cse.msu.edu/Publications/Face/LiaoJain_PartialFR_AlignmentFreeApproach_ICJB11.pdf
Date: 2013
DOI: 10.1109/TPAMI.2012.191
Excerpt: "The proposed MKD-SRC algorithm significantly improves partial face recognition performance. The rank-1 recognition rate of MKD-SRC is 81.31%, while that of SIFT matching is 58.70%."
Confidence: High

#### Entry 20: Deep Softmax Collaborative Representation
Claim: Combining deep VGG features with softmax collaborative representation-based classification significantly improves recognition on seriously degraded faces with noise, blur, and illumination variations.
Source: Engineering Applications of Artificial Intelligence, 2022
URL: https://www.sciencedirect.com/science/article/abs/pii/S0952197620303249
Date: 2022
Excerpt: "Our algorithms are demonstrated to considerably outperform the related state-of-the-art recognition algorithms. We used strategies that target to damage the input image by using Gaussian noise, Gaussian blur, Salt and Pepper noise, illumination, and occlusion."
Confidence: Medium

---

### Quality Scores for Fusion Weight Modulation: Suitability Assessment

| Method | Quality Score Output | Modulates Fusion? | Computational Cost | Suitability for Face+Fingerprint Fusion |
|--------|---------------------|-------------------|-------------------|----------------------------------------|
| AdaFace [^174^] | Feature norm (implicit) | Indirect (loss-based) | Low (no extra forward pass) | Good - norm available during inference |
| MagFace [^177^] | Embedding magnitude | Explicit (comparison score) | Low | Good - magnitude naturally available |
| SER-FIQ [^175^] | Stochastic embedding variance | Explicit quality score | Medium (multiple dropout passes) | Moderate - requires multiple passes |
| CR-FIQA [^199^] | Certainty Ratio (0-1) | Explicit quality score | Medium (separate regression network) | Good - trained quality predictor |
| SDD-FIQA [^201^] | Wasserstein distance-based score | Explicit quality score | Medium | Good - generalizes across FR models |
| GraFIQs [^233^] | Gradient magnitude sum | Explicit quality score | High (backprop required) | Limited - backpropagation overhead |
| SlackedFace [^244^] | P-Norm recognizability index | Explicit quality score | Low | Good - combines norm and proximity |
| QMagFace [^76^] | Quality-aware comparison score | Direct (score-level) | Low | Excellent - designed for fusion |

### Key Takeaways for Face+Fingerprint Fusion

1. **QMagFace** and **AdaFace** provide the most practical quality estimates for fusion weight modulation, as they produce quality indicators with minimal computational overhead.

2. **CR-FIQA** and **SDD-FIQA** offer the most theoretically grounded quality estimates but require additional network training.

3. **SlackedFace's P-Norm** may be particularly valuable for surveillance scenarios where face quality is extremely low, as it specifically distinguishes recognizable from unrecognizable degraded faces.

4. The **Quality-Guided Mixture of Score-Fusion Experts** framework [^144^] provides a concrete architecture for integrating these quality scores into multimodal fusion, demonstrating operational feasibility.

5. **NIST FATE Quality** evaluation results [^226^] provide empirical validation that quality-aware processing reduces FNMR, supporting the operational value of quality-guided fusion.

---

*Document compiled from 20+ independent web searches across academic databases (Google Scholar, arXiv, IEEE Xplore, CVF Open Access, NIST). All sources verified for author-title-venue coherence. Citations marked [CITATION TO VERIFY] where venue assignment requires additional confirmation.*
