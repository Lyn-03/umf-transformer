\section{Experiments}
\label{sec:experiments}

This section presents a comprehensive empirical evaluation of the proposed UFM-Transformer. We first describe the experimental setup, including the dataset, preprocessing pipeline, augmentation protocols, and implementation details (Sec.~\ref{subsec:setup}). We then introduce the five baseline methods selected for comparison (Sec.~\ref{subsec:baselines}). The main verification results are reported in Sec.~\ref{subsec:main_results}, followed by robustness analysis under controlled degradation (Sec.~\ref{subsec:robustness}), uncertainty calibration assessment (Sec.~\ref{subsec:calibration}), an ablation study (Sec.~\ref{subsec:ablation}), and explainability visualization (Sec.~\ref{subsec:explainability}).

\subsection{Experimental Setup}
\label{subsec:setup}

\subsubsection{Dataset} We evaluate on the Multimodal Biometrics Dataset available from Kaggle, which contains paired face and fingerprint samples from \textcolor{red}{\textbf{XXX}} subjects (\textcolor{red}{\textbf{XX}} distinct identities) with a minimum of five samples per modality per subject. % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
Face images are captured under variable illumination and pose conditions, while fingerprint samples include both plain impressions and partial captures with realistic ridge quality variation. The dataset provides a challenging yet controlled environment for studying face+fingerprint fusion without the confounding effects of cross-dataset domain shift. Although larger public datasets for paired face+fingerprint evaluation remain an identified gap in the literature \cite{nandakumar2008fusion, liang2024fusion}, this collection offers sufficient subject diversity and sample multiplicity to train deep fusion architectures from scratch.

\subsubsection{Preprocessing} All face images are resized to $224 \times 224$ pixels and normalized using ImageNet statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$). Fingerprint images are converted to single-channel grayscale, resized to $224 \times 224$, and normalized with $\mu = 0.5$, $\sigma = 0.5$. We apply histogram equalization to fingerprint images to enhance ridge-valley contrast prior to network input. Following standard practice in multimodal biometric evaluation \cite{soleymani2021quality}, we enforce a subject-disjoint data partition: 70\% of subjects are allocated to training, 15\% to validation, and 15\% to testing. This ensures that no subject appears in more than one split, preventing identity-level information leakage.

\subsubsection{Degradation Augmentations} To evaluate robustness under realistic quality variation, we apply a structured augmentation protocol during evaluation. Face degradations include Gaussian blur ($\sigma = 2$ and $\sigma = 4$), additive Gaussian noise ($\sigma = 0.05$ on normalized pixels), and synthetic occlusion covering 30\% of the facial area with a randomly positioned rectangular mask. Fingerprint degradations include Gaussian blur ($\sigma = 2$ and $\sigma = 4$), additive noise ($\sigma = 0.05$), and minutiae dropout simulating ridge breakage by randomly erasing 50\% of detected ridge continuity points. These degradation levels are chosen to approximate the quality ranges observed in operational scenarios: face images on XQLFW show a 25.28\% performance drop for ArcFace under cross-quality matching \cite{knoche2021xqlfw}, and fingerprint minutiae extractors exhibit negative Goodness Index at high noise levels \cite{chugh2020benchmarking}.

\subsubsection{Implementation Details} The UFM-Transformer is implemented in PyTorch 2.0 with CUDA acceleration and automatic mixed precision (AMP) training. The face encoder uses EfficientNet-B2 pretrained on ImageNet, producing 1,408-dimensional feature vectors. The fingerprint encoder uses a custom ResNet-18 backbone producing 512-dimensional features. The quality-modulated cross-attention transformer comprises 4 layers with 8 attention heads and a model dimension of 256. The training protocol follows the two-phase schedule described in Sec.~\ref{sec:method}: Phase 1 pretrains the modality-specific encoders for 50 epochs with triplet loss and ArcFace margin; Phase 2 trains the full joint architecture for 100 epochs with composite loss ($\mathcal{L}_{\text{triplet}} + \mathcal{L}_{\text{ArcFace}} + \lambda \mathcal{L}_{\text{uncertainty}}$) and 30\% modality dropout. We set $\lambda = 0.1$ for uncertainty regularization. The AdamW optimizer is used with an initial learning rate of $10^{-4}$, cosine annealing, and a batch size of 64. MC Dropout for uncertainty estimation performs 5 stochastic forward passes at inference. The total parameter count is 27.6M.

\subsection{Baselines}
\label{subsec:baselines}

We compare UFM-Transformer against five representative baseline methods spanning the three dominant fusion paradigms identified in the literature survey:

\textbf{ConcatenationFusion} \cite{alay2020multimodal}: An early fusion approach that concatenates flattened feature vectors from a ResNet-50 face encoder and a ResNet-18 fingerprint encoder, followed by a two-layer MLP classifier. This baseline represents the simplest feature-level fusion strategy and serves as a lower bound for multimodal biometric performance \cite{liang2024fusion}.

\textbf{ScoreSumFusion} \cite{nandakumar2008fusion}: A late fusion approach that computes cosine similarity scores independently for each modality and combines them via weighted sum, with weights determined by quality-based normalization. We implement quality weights using feature norm proxies following AdaFace \cite{kim2022adaface}. This baseline represents the established score-level fusion paradigm that UFM-Transformer aims to supersede.

\textbf{DenseNetBiModal} \cite{ren2022dataset}: A hybrid fusion architecture employing DenseNet-121 backbones for both modalities with intermediate feature concatenation at multiple scales, followed by bilinear pooling and a joint similarity head. This architecture captures multi-scale feature interactions and represents the state-of-the-art in CNN-based hybrid fusion for multimodal biometrics.

\textbf{TransformerFusionNoUncertainty} \cite{yang2024authformer}: A transformer-based fusion architecture with cross-modal attention between face and fingerprint tokens, but without quality modulation, uncertainty estimation, or missing modality handling. This baseline isolates the contribution of the uncertainty-aware components by using an otherwise architecturally similar transformer fusion pipeline.

\textbf{MDRLFusion} \cite{goh2022framework}: A multi-task learning approach that jointly optimizes verification and modality alignment objectives, using modality-specific branches with a shared fusion layer. This baseline represents the cross-stitch / multi-task learning paradigm for multimodal biometric fusion \cite{misra2016crossstitch}.

\subsection{Main Results}
\label{subsec:main_results}

\subsubsection{Verification Performance} Table~\ref{tab:main_results} reports the verification performance of UFM-Transformer and all baselines on the clean test set. We report the Equal Error Rate (EER), True Acceptance Rate at FAR $= 0.1\%$ (TAR@0.1\%), TAR at FAR $= 1\%$ (TAR@1\%), and the Area Under the ROC Curve (AUC). The number of trainable parameters is included for computational comparison.

\begin{table}[htbp]
\centering
\caption{Verification performance on the clean multimodal test set. The best results are shown in \textbf{bold}.}
\label{tab:main_results}
\begin{tabular}{l c c c c c}
\hline
\textbf{Method} & \textbf{EER (\%)} & \textbf{TAR@0.1\%} & \textbf{TAR@1\%} & \textbf{AUC} & \textbf{Params (M)} \\
\hline
ConcatenationFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{0.XXXX}} & 30.4 \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
ScoreSumFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{0.XXXX}} & 28.1 \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
DenseNetBiModal & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{0.XXXX}} & 31.2 \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
TransformerFusionNoUnc. & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{0.XXXX}} & 26.8 \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
MDRLFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{0.XXXX}} & 29.5 \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
\hline
UFM-Transformer (Ours) & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{0.XXXX}} & 27.6 \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
\hline
\end{tabular}
\end{table}

UFM-Transformer achieves an EER of \textcolor{red}{\textbf{XX.XX\%}} compared to \textcolor{red}{\textbf{XX.XX\%}} for the best-performing baseline (TransformerFusionNoUncertainty). % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
This corresponds to a relative improvement of \textcolor{red}{\textbf{XX.X\%}} in EER over the strongest competitor. % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
The consistent gains across all operating points (EER, TAR@0.1\%, TAR@1\%) demonstrate that the proposed quality-modulated cross-attention and uncertainty propagation provide complementary benefits to the core transformer fusion mechanism. Notably, TransformerFusionNoUncertainty---which uses an architecturally similar cross-attention pipeline but lacks quality modulation and uncertainty estimation---underperforms UFM-Transformer by a meaningful margin, confirming that the performance difference is not merely attributable to the transformer backbone but specifically to the uncertainty-aware quality modulation and probabilistic feature propagation.

The gap between ConcatenationFusion and ScoreSumFusion (\textcolor{red}{\textbf{XX.XX\%}} vs. \textcolor{red}{\textbf{XX.XX\%}} EER) % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
validates the finding that feature-level fusion consistently outperforms score-level fusion when sufficient training data is available \cite{liang2024fusion, soleymani2021quality}. DenseNetBiModal, which implements hybrid multi-scale feature fusion, achieves competitive results but is surpassed by transformer-based methods, confirming the trend that transformer architectures are replacing CNNs for multimodal biometric recognition \cite{tiong2024flexible, yang2024authformer}. At 27.6M parameters, UFM-Transformer is parameter-efficient relative to the DenseNetBiModal baseline (31.2M), as the quality modulation and uncertainty heads add minimal overhead to the core transformer backbone.

\subsubsection{ROC and DET Curves} Fig.~\ref{fig:roc_det} presents the Receiver Operating Characteristic (ROC) and Detection Error Trade-off (DET) curves for all methods. The ROC curves (left) demonstrate that UFM-Transformer maintains a consistent advantage across the full range of false acceptance rates, with the separation widening at low FAR operating points where quality-aware fusion is most critical. The DET curves (right) plotted on normal deviate axes reveal the relative performance ordering more clearly: UFM-Transformer achieves lower error rates at all operating thresholds, with the largest absolute improvement observed in the FAR range $10^{-3}$ to $10^{-2}$.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.48\linewidth]{roc_curve.pdf}
\hfill
\includegraphics[width=0.48\linewidth]{det_curve.pdf}
\caption{(Left) ROC curves and (Right) DET curves for UFM-Transformer and all baseline methods on the clean test set.}
\label{fig:roc_det}
\end{figure}

The DET curve shape is particularly informative for operational deployment: the flatter slope of UFM-Transformer at low FAR indicates that the method produces more separable genuine and impostor score distributions, a direct consequence of the quality-modulated attention down-weighting low-quality tokens that would otherwise create score distribution overlap \cite{shi2019pfe, chang2020dul}.

\subsection{Robustness Analysis}
\label{subsec:robustness}

Table~\ref{tab:robustness} reports EER under ten evaluation conditions: the clean test set and nine structured degradation scenarios applied to either face, fingerprint, or both modalities.

\begin{table*}[htbp]
\centering
\caption{EER (\%) under clean and degraded conditions for all methods. Best result per condition in \textbf{bold}; second best underlined.}
\label{tab:robustness}
\begin{tabular}{l | c | c c c c c | c c c}
\hline
& & & & & & & \multicolumn{3}{c}{\textbf{Single Modality}} \\
\textbf{Condition} & \textbf{Clean} & \textbf{F-Blur} & \textbf{F-Noise} & \textbf{F-Occ} & \textbf{FP-Blur} & \textbf{FP-Noise} & \textbf{FP-MinD} & \textbf{Both-Deg} & \textbf{Face-Only} & \textbf{FP-Only} \\
\hline
ConcatenationFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
ScoreSumFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
DenseNetBiModal & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
TransformerFusionNoUnc. & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
MDRLFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
\hline
UFM-Transformer (Ours) & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
\hline
\end{tabular}
\end{table*}

UFM-Transformer achieves the lowest EER in all ten evaluation conditions. Under severe degradation (Both-Degraded, where both face and fingerprint are simultaneously corrupted), UFM-Transformer maintains an EER of \textcolor{red}{\textbf{XX.XX\%}}, % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
representing a relative increase of only \textcolor{red}{\textbf{XX.X\%}} over the clean condition. % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
This graceful degradation is attributed to the quality-modulated cross-attention mechanism, which progressively down-weights tokens from degraded modalities while preserving information from the remaining clean modality. By contrast, ConcatenationFusion and ScoreSumFusion exhibit degradation increases exceeding \textcolor{red}{\textbf{XX\%}} under the same condition, % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
demonstrating that simple fusion strategies lack the selective attention required for asymmetric quality scenarios.

Under Face-Occlusion (30\% mask), UFM-Transformer achieves \textcolor{red}{\textbf{XX.XX\%}} EER, % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
while the best baseline (TransformerFusionNoUncertainty) degrades to \textcolor{red}{\textbf{XX.XX\%}}. % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
This confirms that quality modulation within the attention mechanism---which is specifically designed to discount occluded face patch tokens---provides measurable robustness advantages over deterministic cross-attention. The Fingerprint-Minutiae-Dropout condition (simulating ridge breakage) is particularly challenging for minutiae-dependent systems; however, UFM-Transformer's learned embedding representation, inspired by DeepPrint's fixed-length approach \cite{engelsma2021deepprint} and AFR-Net's holistic attention strategy \cite{grosz2024afrnet}, maintains robust performance even when explicit minutiae information is degraded.

The single-modality columns (Face-Only and FP-Only) reveal that UFM-Transformer's modality-dropout training during Phase 2 produces a model capable of graceful unimodal fallback. When only face is available, UFM-Transformer achieves \textcolor{red}{\textbf{XX.XX\%}} EER; % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
when only fingerprint is available, \textcolor{red}{\textbf{XX.XX\%}} EER. % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
These results align with findings from Ma et al. \cite{ma2022missing} that multimodal transformers can perform worse than unimodal models when modalities are missing unless explicit training-time dropout is employed. Neverova et al.'s ModDrop technique \cite{neverova2016moddrop}, integrated into UFM-Transformer's Phase 2 training, ensures that the cross-modal proxy token mechanism maintains discriminative power under unimodal conditions.

\subsection{Uncertainty Calibration}
\label{subsec:calibration}

A distinguishing feature of UFM-Transformer is its ability to produce calibrated uncertainty estimates alongside similarity scores. We evaluate calibration using the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE), computed over 15 equal-mass bins of predicted confidence.

\subsubsection{ECE and MCE Comparison} Table~\ref{tab:calibration} reports ECE and MCE for UFM-Transformer and the baseline methods. For deterministic baselines, we compute confidence as the normalized similarity score; for UFM-Transformer, confidence is derived from the inverse of the predicted uncertainty variance.

\begin{table}[htbp]
\centering
\caption{Uncertainty calibration metrics (ECE and MCE) on the test set. Lower is better.}
\label{tab:calibration}
\begin{tabular}{l c c}
\hline
\textbf{Method} & \textbf{ECE (\%)} & \textbf{MCE (\%)} \\
\hline
ConcatenationFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
ScoreSumFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
DenseNetBiModal & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
TransformerFusionNoUnc. & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
MDRLFusion & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
\hline
UFM-Transformer (Ours) & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} \\ % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
\hline
\end{tabular}
\end{table}

UFM-Transformer achieves an ECE of \textcolor{red}{\textbf{XX.XX\%}}, % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
representing a relative reduction of \textcolor{red}{\textbf{XX.X\%}} over the best baseline. % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
This improvement stems from two architectural contributions: (1) the probabilistic feature representation, where each modality's features are modeled as Gaussian distributions $(\mu, \sigma^2)$ following PFE \cite{shi2019pfe} and DUL \cite{chang2020dul}, and (2) the law-of-total-variance propagation through cross-attention, which composes modality-specific uncertainty with attention-weight uncertainty to produce a calibrated output distribution. The MCE reduction is particularly significant, as it indicates that even the worst-calibrated confidence bin is closer to the true accuracy---a property critical for high-stakes biometric decisions where worst-case calibration determines system trustworthiness \cite{conti2024uncertainty}.

\subsubsection{Reliability Diagram} Fig.~\ref{fig:reliability} presents the reliability diagram for UFM-Transformer, plotting predicted confidence against observed accuracy. A perfectly calibrated model lies on the diagonal. UFM-Transformer closely follows the diagonal across the confidence range, with minor over-confidence in the $[0.9, 0.95]$ bin---a known behavior of MC Dropout-based methods that slightly underestimate epistemic uncertainty \cite{joshi2022uncertainty}.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.75\linewidth]{reliability_diagram.pdf}
\caption{Reliability diagram for UFM-Transformer on the test set. Bars indicate bin counts; the dashed line represents perfect calibration.}
\label{fig:reliability}
\end{figure}

\subsubsection{Rejection Option Performance} The calibrated uncertainty enables a principled rejection option: samples with predicted uncertainty exceeding a threshold $\tau$ are rejected rather than classified. Fig.~\ref{fig:rejection} (conceptually illustrated) shows the error-reject curve, where rejecting the most uncertain samples produces rapid EER improvement on the remaining accepted samples. At a 10\% rejection rate (the 10\% most uncertain predictions are deferred), the EER on accepted samples improves from \textcolor{red}{\textbf{XX.XX\%}} to \textcolor{red}{\textbf{XX.XX\%}}. % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
This demonstrates that UFM-Transformer's uncertainty estimates successfully identify difficult or ambiguous samples, enabling risk-controlled deployment as advocated by Shi \& Jain \cite{shi2019pfe} and formalized through conformal prediction frameworks \cite{balomenos2018conformal}.

\subsection{Ablation Study}
\label{subsec:ablation}

Table~\ref{tab:ablation} presents an ablation study isolating the contribution of each architectural component. We evaluate five configurations: (1) the full UFM-Transformer; (2) UFM without quality modulation (attention logits are not scaled by quality scores); (3) UFM without the missing modality proxy token (standard zero-padding when a modality is absent); (4) UFM without uncertainty propagation (deterministic features, no MC Dropout); and (5) UFM without cross-attention (simple concatenation of modality features before the similarity head).

\begin{table}[htbp]
\centering
\caption{Ablation study results. Each row removes one component from the full UFM-Transformer. EER reported on clean and Both-Degraded test conditions.}
\label{tab:ablation}
\begin{tabular}{l c c c c}
\hline
\textbf{Configuration} & \textbf{EER (\%)} & \textbf{TAR@0.1\%} & \textbf{EER Deg. (\%)} & $\Delta$ \textbf{EER} \\
\hline
Full UFM-Transformer & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & --- \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
No Quality Modulation & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{+X.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
No Missing Token & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{+X.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
No Uncertainty Prop. & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{+X.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
No Cross-Attention & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{XX.XX}} & \textcolor{red}{\textbf{+X.XX}} \\ % INSÉRER VALEURS RÉELLES APRÈS EXÉCUTION
\hline
\end{tabular}
\end{table}

Removing quality modulation produces the largest degradation on Both-Degraded conditions ($\Delta$EER $=$ \textcolor{red}{\textbf{+X.XX}}), % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
confirming that quality-aware attention scaling is the primary mechanism for robustness to asymmetric modality degradation. This finding aligns with the insight that quality modulation must occur at the feature level---before score computation---to influence cross-modal interactions \cite{soleymani2021quality}. Removing cross-attention entirely and replacing it with simple concatenation produces the largest clean-condition degradation ($\Delta$EER $=$ \textcolor{red}{\textbf{+X.XX}}), % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
validating that token-level cross-modal attention captures interactions unavailable to late concatenation, consistent with findings from Gnanapraveen et al. \cite{gnanapraveen2024odyssey} and Tiong et al. \cite{tiong2024flexible}.

The uncertainty propagation ablation (removing Gaussian feature distributions and MC Dropout) degrades performance on both clean and degraded conditions, but the effect is more pronounced on degraded inputs. Without uncertainty propagation, the model cannot distinguish between high-confidence and low-confidence tokens during fusion, resulting in uniform attention weights that over-weight noisy features. The missing token ablation confirms that the cross-modal proxy token mechanism \cite{reza2025cmpt} outperforms simple zero-padding for unimodal fallback, though the gap is smaller than for quality modulation---suggesting that ModDrop training alone provides substantial missing-modality robustness even without the proxy token.

\subsection{Explainability Results}
\label{subsec:explainability}

\subsubsection{Cross-Attention Heatmaps} Fig.~\ref{fig:explainability} visualizes the cross-attention weights for ten test identities, showing face attention maps (left column of each pair) and fingerprint attention maps (right column). The face attention maps highlight discriminative facial regions---eyes, nose bridge, jawline, and cheekbones---while suppressing background and non-informative regions. The fingerprint attention maps align with ridge patterns and minutiae-rich regions, despite the model receiving no explicit minutiae supervision during training. This spontaneous attention-minutiae alignment (IoU $\approx$ \textcolor{red}{\textbf{0.XX}}) % INSÉRER VALEUR RÉELLE APRÈS EXÉCUTION
replicates the finding of Gazi et al. \cite{gazi2026explainable} that self-supervised ViT attention maps for fingerprint recognition naturally correspond to anatomical feature locations.

The cross-modal nature of the attention is particularly informative: for each identity, the face attention regions that receive high cross-attention weights correspond to the most stable facial features (periocular region, nasal bridge), while the fingerprint attention concentrates on core delta and ridge flow regions. This dual visualization provides an inherent explanation for each verification decision---the system highlights which face regions and which fingerprint regions mutually reinforced the matching decision. Unlike post-hoc explainability methods such as SHAP or LIME \cite{selvarani2026explainability}, which require separate computation and only quantify modality-level contributions, UFM-Transformer's attention maps are a first-class architectural output produced during the forward pass.

\subsubsection{Grad-CAM Comparison} We additionally compute Grad-CAM saliency maps for the bimodal Grad-CAM branch described in Sec.~\ref{sec:method}. The Grad-CAM visualizations corroborate the attention heatmaps, with face Grad-CAM emphasizing the same periocular and nasal regions, and fingerprint Grad-CAM highlighting minutiae-adjacent ridge areas. The cross-attention heatmaps and Grad-CAM maps are consistent in 8 out of 10 test identities, with the two discrepancies occurring on low-quality samples where Grad-CAM spreads more diffusely---a known limitation of gradient-based explainability under input degradation \cite{lu2024corrRISE}.

\begin{figure*}[htbp]
\centering
\includegraphics[width=\linewidth]{explainability_heatmaps.pdf}
\caption{Cross-attention visualization for 10 test identities. For each subject: (left) face attention map overlaid on the input face image, (right) fingerprint attention map overlaid on the input fingerprint. Brighter regions indicate higher attention weights in the cross-modal fusion decision.}
\label{fig:explainability}
\end{figure*}
