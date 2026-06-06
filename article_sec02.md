\section{Related Work}

\subsection{Deep Multimodal Fusion Architectures}

\subsubsection{Early fusion: feature concatenation and joint embedding learning}

Feature-level fusion remains the dominant paradigm in deep multimodal biometric systems. Soleymani et al. demonstrated that multi-level feature abstraction---extracting representations from multiple convolutional layers of modality-specific CNNs and jointly optimizing them---outperforms single-layer fusion, achieving 99.34% rank-one accuracy on the BioCop database for face+iris+fingerprint identification [^3^]. The same authors showed that Generalized Compact Bilinear (GCB) fusion, which captures multiplicative interactions between modality features via tensor sketch projection, attains state-of-the-art results on CMU Multi-PIE and BIOMDATA datasets with significant parameter reduction [^2^]. Talreja et al. confirmed that bilinear (multiplicative) fusion consistently outperforms linear concatenation, reporting an EER of 0.84% on face+iris with a bilinear architecture [^11^]. Misra et al. introduced cross-stitch networks that learn adaptive linear combinations of layer outputs to enable shared representation learning across modalities [^4^], later extended by Ruder et al. through sluice networks, achieving 12.8% error reduction over single-task models [^5^].

A critical vulnerability of early fusion is that when one modality is degraded, corrupted features propagate directly into the joint representation, compromising the entire fused embedding [^6^]. Unlike score-level alternatives, early fusion lacks a mechanism to suppress degraded modality contributions. Recent attention-based gating mechanisms attempt to mitigate this issue---for instance, Wang et al. proposed a Channel Spatial Attention Fusion Module that dynamically adjusts fusion weights in channel and spatial dimensions---yet the problem remains largely unsolved for face+fingerprint fusion specifically [^9^].

\subsubsection{Late fusion: score-level combination}

Score-level fusion continues to dominate deployed multimodal biometric systems due to its modularity. Jain et al. systematically compared five score normalization techniques across face, fingerprint, and hand-geometry modalities, establishing that z-score normalization followed by the sum rule achieves the best generalization [^1^]. Nandakumar et al. extended this framework through a likelihood ratio approach that models genuine and impostor score distributions as finite Gaussian mixture models, achieving consistently high performance across multiple databases [^4^].

Quality-aware score fusion marked a significant conceptual advance. Nandakumar et al. first demonstrated that incorporating sample quality measures into likelihood ratio-based fusion improves recognition by dynamically weighting modalities based on reliability [^5^]. QMagFace (WACV 2023) exploits the linearity between embedding norms and comparison scores, achieving 98.74% on CFP-FP under quality-aware scoring [^21^]. Alay and Al-Baity reported 100% accuracy with score-level fusion (arithmetic mean rule) versus 99.39% with feature-level fusion on SDUMLA-HMT, attributing this to fixed softmax classifier behavior rather than inherent superiority [^7^].

The fundamental limitation of score-level fusion is irreversible information loss: by the time scores are produced, the rich feature-space topology of cross-modal interactions has been discarded. As Liang et al. observed in their comprehensive ACM Computing Surveys analysis, decision-level fusion "provides limited interpretability of the multimodal interactions" and its performance is "limited by one modality" [^11^].

\subsubsection{Hybrid fusion: multi-stage pipelines}

Hybrid strategies that combine feature-level and score-level fusion have emerged as the consensus best approach. Aswin et al. demonstrated that a 70% score / 30% feature hybrid achieves 99.79% accuracy with an EER of 0.18% on the NUPT-FPV dataset, surpassing pure score-level (99.37%) and pure feature-level alternatives [^3^]. Transformer-based hybrid architectures represent the current frontier. Nagrani et al. introduced the Multimodal Bottleneck Transformer (MBT), which forces information between modalities to pass through bottleneck latents, reducing computational cost while maintaining performance [^106^]. Lu et al. proposed Multilevel Parallel Attention Knowledge Distillation (MPAD), integrating a Parallel Attention Fusion Module across multiple network levels to enable lightweight student networks [^54^]. These hybrid approaches, however, incur significant computational complexity: they require separate modality-specific encoders, intermediate fusion modules, and score-level combination stages, each with their own hyperparameters.

\subsection{Attention and Transformer Mechanisms for Biometrics}

\subsubsection{Self-attention for face recognition}

The application of Vision Transformers to biometric recognition has advanced rapidly. TransFace (Dan et al., ICCV 2023) identified critical training incompatibilities when applying ViTs to face recognition, proposing data augmentation and hard sample mining strategies that improved ViT-S performance by +0.56% on IJB-C at TAR@FAR=$10^{-4}$ [^2^]. LVFace (You et al., 2025) introduced Progressive Cluster Optimization (PCO), achieving 97.25% TAR@FAR=$10^{-5}$ on IJB-C with a ViT-L backbone [^3^]. ViT-GCT (Chettaoui et al., ICLR 2026) demonstrated that a Global Context Token (GCT) replacing the standard CLS token improves performance by +1.0% on IJB-C at FAR=$10^{-5}$ [^4^]. For fingerprint recognition, AFR-Net (Grosz and Jain, IEEE T-BIOM 2024) combined CNN-based and ViT embeddings, achieving 99.78% rank-1 retrieval on NIST SD14 against a 100K gallery [^9^].

\subsubsection{Cross-modal attention: quality-gated fusion and bottleneck attention}

Cross-modal attention has shown consistent advantages over self-attention and simple concatenation. Tiong et al. (CVPR 2024) proposed Multimodal Fusion Attention (MFA) within a ViT architecture, achieving state-of-the-art on face-periocular recognition with Multimodal Prompt Tuning (MPT) for parameter-efficient modality alignment [^1^]. Gnanapraveen et al. (Odyssey 2024) provided empirical evidence that cross-attention fusion achieves an EER of 2.387, compared to 2.412 for self-attention, 2.489 for feature concatenation, and 2.521 for score-level fusion on audio-visual person recognition [^3^]. Soleymani et al. (IEEE TBIOM 2021) showed that quality-aware fusion with weakly-supervised quality scores outperforms rank- and score-level fusion by more than 30% TAR at FAR=$10^{-4}$ [^2^]. AuthFormer (Yang et al., 2024) combined cross-attention with Gated Residual Networks, achieving 99.73% accuracy on face+fingerprint+voice authentication with only 2 encoder layers [^8^].

\subsubsection{Gap: no true inter-modal cross-attention for face+fingerprint verification}

Despite extensive work on face+periocular, face+voice, and face+iris fusion, a critical gap remains: no published work implements true inter-modal cross-attention specifically for face+fingerprint biometric verification, where face patch tokens and fingerprint ridge tokens directly attend to each other within a shared transformer space. Current approaches such as Tiny Transformers [^10^] and CSMG [^11^] use separate modality-specific transformer backbones with quality-gated score-level fusion rather than token-level cross-modal attention. As Liang et al. conclude, existing systems "exchange cross-modal information in the last layers or penultimate layers of decoders," fundamentally limiting cross-modal interaction [^11^].

\subsection{Robustness and Missing Modality Handling}

\subsubsection{Quality-aware recognition: face and fingerprint}

Quality-aware recognition has become essential for real-world deployment. For face recognition, MagFace (CVPR 2021) demonstrated that a magnitude-aware angular margin loss simultaneously learns discriminative features and quality estimates [^34^]. CR-FIQA (CVPR 2023) learned face image quality assessment by relative classifiability [^21^]. For fingerprint recognition, NFIQ 2.0 provides standardized quality metrics [^45^], while Priesnitz et al. (BIOSIG 2025) demonstrated that deep learning quality assessment outperforms NFIQ 2.3 by 12.29% [^44^]. QMagFace (WACV 2023) achieves 98.50% on AgeDB and 98.74% on CFP-FP under quality-aware scoring [^21^].

\subsubsection{Missing modality in medical imaging versus biometrics}

Missing modality handling is well-studied in medical imaging but critically under-addressed in biometrics. Ma et al. (AAAI 2021) pioneered Bayesian meta-regularization for severely missing modalities (up to 90% missing ratio) with SMIL [^6^]. Lee et al. (CVPR 2023) introduced missing-aware prompt tuning, requiring less than 1% learnable parameters [^5^]. Reza et al. (2025) proposed Cross-Modal Proxy Tokens (CMPTs), which approximate missing modality class tokens by attending to available modality tokens, outperforming 12 recent baselines [^4^]. Zhang et al. (CVPR 2023) enabled arbitrary-modal inference through CMNeXt, scaling from 1 to 81 modalities [^10^].

Ma et al. (CVPR 2022) found that multimodal transformers are NOT naturally robust to missing modalities---they can perform worse than unimodal models [^3^]. As the comprehensive survey by Wu et al. (TMLR 2026) confirms, the intersection of missing-modality research and biometric fusion is essentially empty: out of 17 surveyed methods, only 3 are biometric-specific, and all use non-deep-learning approaches [^1^].

\subsubsection{Modality dropout, learnable tokens, and hallucination networks}

Modality dropout (ModDrop), introduced by Neverova et al. (T-PAMI 2016), remains the foundational training-time technique: randomly masking modality channels forces networks to learn cross-modal correlations [^2^]. Shen et al. (2025) extended this through Multimodal Adaptive Dropout (MAD), dynamically discarding features from converged modalities during training [^12^]. Hallucination networks (Hoffman et al., CVPR 2016) predict missing modality features from available ones via L2 feature alignment [^16^]. Despite demonstrated effectiveness in other domains, none of these techniques have been applied to face+fingerprint biometric fusion.

\subsection{Uncertainty Quantification in Biometric Verification}

\subsubsection{Probabilistic face embeddings}

Probabilistic Face Embeddings (PFE), introduced by Shi and Jain (ICCV 2019), represent each face image as a Gaussian distribution $\mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ in latent space, where $\boldsymbol{\mu}$ estimates the identity feature and $\boldsymbol{\sigma}$ quantifies uncertainty. The Mutual Likelihood Score (MLS) replaces cosine similarity for matching, providing calibrated confidence estimates [^1^]. Data Uncertainty Learning (DUL) extended PFE by jointly learning identity and uncertainty end-to-end, achieving 90.23% TPR@FPR=$10^{-5}$ on IJB-C compared to 83.06% for the deterministic baseline [^2^]. Holistic Uncertainty Estimation (HolUE) combines gallery-aware uncertainty with sample quality through a Bayesian model, outperforming PFE for open-set recognition [^6^].

\subsubsection{Monte-Carlo Dropout and Deep Ensembles for biometric confidence}

Monte Carlo Dropout (MC-Dropout) has been applied to fingerprint enhancement for model uncertainty estimation. Joshi et al. introduced MU-GAN and DU-GAN, demonstrating that model uncertainty is higher at incorrectly predicted pixels while data uncertainty is higher on noisy input regions [^3^]. Conti et al. (ICLR 2024) proved that naive bootstrap produces invalid confidence bands for biometric ROC curves because FAR/FRR are U-statistics; their recentered bootstrap provides valid inference about performance and fairness metrics [^5^]. Bayesian Metric Learning (LAM, NeurIPS 2023) uses Laplace approximation over network weights, yielding better-calibrated uncertainties and state-of-the-art OOD detection with lower Expected Calibration Error [^7^].

\subsubsection{Gap: no uncertainty-aware multimodal biometric system}

Despite rich literature on both uncertainty quantification for face recognition and multimodal biometric fusion, the intersection is essentially empty. Dempster-Shafer theory has been applied to score-level multimodal fusion with explicit uncertainty masses [^10^], and context-aware decision fusion addresses uncertainty at the decision level [^11^], but no framework propagates uncertainty from each modality through a cross-modal attention mechanism to produce calibrated bimodal confidence scores. He et al. (ACM Computing Surveys 2025) confirm that biometric applications of deep learning uncertainty remain under-studied [^12^].

\subsection{Explainability in Biometric Systems}

\subsubsection{Grad-CAM for face recognition, attention maps for fingerprints}

Grad-CAM and its variants have been widely adopted for explaining face recognition decisions, but produce less stable maps than newer perturbation-based methods. CorrRISE (Lu et al., WACV 2024) outperforms Grad-CAM, Grad-CAM++, LIME, RISE, and xFace in quantitative benchmarks, achieving Deletion scores of 23.29% versus 50.65% for Grad-CAM on LFW [^316^]. xSSAB (Huber et al., WACV 2024) provides a training-free white-box approach that backpropagates the cosine similarity score through a Siamese network [^389^]. For fingerprint recognition, Tan and Kumar (IEEE T-IFS, 2021) introduced a minutiae attention network that generates minutiae likelihood maps serving as attention maps for contactless-to-contact identification [^315^].

\subsubsection{Gap: no unified bimodal explanation (face regions + minutiae simultaneously)}

No published work provides unified visual explanations simultaneously highlighting which facial regions AND which fingerprint minutiae points contributed to a face+fingerprint biometric decision. Selvarani and Rani (ICAISDA 2025) quantified modality-level contributions via SHAP (Face 48%, Fingerprint 31%, Palmprint 21%) with 91.7% average local fidelity using LIME, but their analysis remains at the modality level rather than generating fused visual saliency maps [^56^]. Gazi et al. (2026) demonstrated that self-supervised DINOv2 ViTs spontaneously learn attention maps aligning with minutiae-rich regions (IoU 0.41$\pm$0.07), providing forensically meaningful explanations without explicit minutiae supervision [^48^]. As Neto et al. observed in their comprehensive survey, biometric explainability remains fragmented across modalities without unified frameworks [^344^].

\subsection{Positioning Against Prior Work}

Table~\ref{tab:comparison} summarizes the architectural and functional differences between UFM-Transformer and the six most relevant prior works across eight evaluation dimensions. No existing method satisfies all four desiderata for trustworthy multimodal biometric verification.

\begin{table}[htbp]
\centering
\caption{Comparison of UFM-Transformer with six closest prior works. For EER, results are reported as published on the respective evaluation dataset; direct cross-dataset comparison is not meaningful.}
\label{tab:comparison}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccccc}
\hline
\textbf{Method} & \textbf{Fusion Type} & \textbf{Quality-} & \textbf{Missing} & \textbf{Uncertainty} & \textbf{Explainable} & \textbf{\#Params} & \textbf{EER (\%)} \\
& & \textbf{Aware} & \textbf{Modality} & & & & \\
\hline
Soleymani et al. [^2^] & Feature + Bilinear & Weakly-sup. & \ding{55} & \ding{55} & \ding{55} & $\sim$20M & 0.84 \\
Tiny Transformers [^10^] & Score + Gating & Laplacian MLP & \ding{55} & \ding{55} & \ding{55} & $<$3M & 11.7$^\dagger$ \\
AuthFormer [^8^] & Cross-attention + GRN & \ding{55} & Adaptive$^*$ & \ding{55} & \ding{55} & $\sim$10M & N/A \\
Tiong et al. (MFA-ViT) [^1^] & Cross-attention & \ding{55} & \ding{55} & \ding{55} & Grad-CAM & $\sim$86M & N/A \\
PFE [^1^] & Unimodal (face) & \ding{55} & N/A & Gaussian emb. & \ding{55} & $\sim$25M & N/A \\
QME [^70^] & MoE (score) & Quality-guided & Partial$^{**}$ & \ding{55} & \ding{55} & $\sim$50M & N/A \\
\hline
\textbf{UFM-Transformer} & \textbf{Cross-attn} & \textbf{Token-level} & \textbf{ModDrop} & \textbf{Gaussian} & \textbf{Attention} & $\mathbf{\sim}$\textbf{30M} & \textcolor{red}{\textbf{XX.XX}} \\
& \textbf{+ Bilinear} & \textbf{quality gate} & \textbf{+ CMPT} & \textbf{propagation} & \textbf{maps} & & \textbf{(\%)} \\
\hline
\end{tabular}
}
\begin{tablenotes}
\item $^\dagger$EER without quality gating (TAR drops from 95.0\% to 88.6\%). $^*$Adaptive module adjusts to input modality count but not sensor failure. $^{**}$Handles missing modalities through dynamic expert weighting for face+gait+body, not face+fingerprint.
\end{tablenotes}
\end{table}

The closest prior work is Tiny Transformers (PeerJ 2026) [^10^], which combines Tiny ViT and Swin Transformer with quality-gated score fusion for iris+fingerprint authentication. Their quality-gating mechanism uses a 2-layer MLP that takes Laplacian sharpness as input, achieving inference latency below 50ms with fewer than 3M parameters. However, Tiny Transformers has three critical limitations: (i) quality gating is applied at the score level, which cannot recover feature-space cross-modal information lost during independent encoding; (ii) it lacks any mechanism for handling missing modalities at inference time; and (iii) it provides no uncertainty estimates or bimodal explainability.

The second closest work is AuthFormer (Yang et al., arXiv 2024) [^8^], which proposes an adaptive multimodal biometric authentication transformer using cross-attention and Gated Residual Networks for face+fingerprint+voice. AuthFormer achieves 99.73% accuracy with only 2 encoder layers and flexibly adjusts to varying numbers of input modalities. Its primary limitation is that it remains deterministic without quality-aware attention weighting, uncertainty quantification, or intrinsic explainability.

The third closest work is QME---Quality-Guided Mixture of Score-Fusion Experts (Zhu et al., ICCV 2025) [^70^], which presents a learnable score-fusion strategy using MoE for whole-body biometric recognition. QME handles missing modalities through dynamic expert weighting and uses quality estimates to gate expert contributions. Its primary limitation is that it operates at the score level rather than the feature level, does not propagate uncertainty through the fusion mechanism, and was evaluated on face+gait+body rather than face+fingerprint.

Table~\ref{tab:comparison} confirms that UFM-Transformer is the first architecture to simultaneously address all four requirements for trustworthy multimodal face+fingerprint biometric verification. Section~\ref{sec:method} describes the UFM-Transformer architecture in detail, beginning with the modality-specific ViT encoders, the cross-modal attention mechanism with quality modulation, the missing modality handling strategy, the uncertainty propagation framework, and the bimodal explainability mechanism.
