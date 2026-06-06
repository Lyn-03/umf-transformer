\section{Proposed Method}
\label{sec:method}

This section presents the UFM-Transformer (Uncertainty-aware Fusion Multimodal Transformer), a unified architecture for face+fingerprint biometric verification. The design is motivated by four requirements identified in the research gap analysis: (i) cross-modal attention with per-token quality modulation, (ii) native missing-modality handling, (iii) uncertainty-calibrated similarity scores, and (iv) bimodal explainability. We first formalize the verification problem, then describe the architecture in a bottom-up manner from modality-specific encoders through the quality-modulated cross-attention transformer to the similarity and uncertainty heads.

\subsection{Problem Formulation}
\label{sec:problem_formulation}

\subsubsection{Notation and Input Space}

Let $\mathcal{X}_f = \mathbb{R}^{3 \times H_f \times W_f}$ denote the space of face images and $\mathcal{X}_p = \mathbb{R}^{3 \times H_p \times W_p}$ the space of fingerprint images, with standard resolutions $H_f = W_f = 224$ and $H_p = W_p = 224$. Each training sample consists of a tuple $(x_f, x_p, y, q_f, q_p, m_f, m_p)$ where $x_f \in \mathcal{X}_f$ is the face image, $x_p \in \mathcal{X}_p$ is the fingerprint image, and $y \in \{1, \dots, N_{id}\}$ is the identity label over $N_{id}$ subjects. The quality scores $q_f, q_p \in [0, 1]$ quantify the biometric information content of each modality; these may be obtained from lightweight learned estimators or external quality assessment modules (SER-FIQ \cite{terhorst2020serfiq} for face, NFIQ 2.0 \cite{tabassi2021nist} for fingerprint). The binary mask vector $\mathbf{m} = (m_f, m_p) \in \{0, 1\}^2$ indicates modality availability at inference: $m_i = 1$ if modality $i$ is present and $m_i = 0$ if it is missing due to sensor failure or user non-cooperation.

\subsubsection{Open-Set Verification Objective}

Following the standard open-set protocol \cite{nandakumar2008likelihood}, the model must learn a similarity function $s: \mathcal{X}_f \times \mathcal{X}_p \times \mathcal{X}_f \times \mathcal{X}_p \rightarrow \mathbb{R}$ that assigns high scores to genuine pairs (same identity) and low scores to impostor pairs (different identities). Crucially, UFM-Transformer extends this to output a calibrated confidence interval:
\begin{equation}
\label{eq:verification_objective}
s(\mathbf{x}, \mathbf{x}') \pm \kappa \cdot \sigma_{\text{total}}(\mathbf{x}, \mathbf{x}'),
\end{equation}
where $\mathbf{x} = (x_f, x_p)$ denotes the bimodal probe, $\mathbf{x}' = (x_f', x_p')$ the bimodal gallery template, $\sigma_{\text{total}}$ the total uncertainty estimate, and $\kappa$ a user-defined confidence multiplier. The interval formulation enables a principled rejection option: when the interval spans the decision threshold, the system abstains from classification \cite{balomenos2018automatic}.

\subsection{Architecture Overview}
\label{sec:architecture_overview}

Figure \ref{fig:architecture} depicts the complete UFM-Transformer pipeline. The architecture comprises four stages: (1) \textit{modality-specific encoders} that extract spatial feature maps from each biometric input; (2) \textit{common projectors} that map features to a shared 256-dimensional hypersphere, with a learnable missing-modality token for absent inputs; (3) a \textit{quality-modulated cross-attention transformer} (QMCAT) that performs bidirectional cross-modal attention between face and fingerprint tokens, modulated by per-modality quality scores; and (4) \textit{similarity and uncertainty heads} that produce a cosine similarity score with ArcFace margin and a Monte-Carlo uncertainty estimate. The entire pipeline is trained end-to-end with a composite loss comprising triplet, ArcFace, and uncertainty terms.

\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{architecture_ufm.pdf}
\caption{\textbf{UFM-Transformer architecture.} Face and fingerprint images are processed by modality-specific encoders ($\mathcal{E}_f$: EfficientNet-B2, $\mathcal{E}_p$: ResNet-style CNN) to produce feature maps $f_f \in \mathbb{R}^{B \times 1408 \times 7 \times 7}$ and $f_p \in \mathbb{R}^{B \times 512 \times 7 \times 7}$. Lightweight quality estimators predict $q_f, q_p \in [0,1]$. Spatial features are flattened, L2-normalized, and projected to 256-D vectors. Missing modalities are replaced by a learnable token $\mathbf{t}_{\text{miss}}$. The Quality-Modulated Cross-Attention Transformer (QMCAT) performs bidirectional cross-attention with quality scaling. The similarity head outputs $\hat{s}$ via ArcFace-margin cosine similarity; the uncertainty head estimates $\sigma_{\text{total}}$ via MC Dropout variance decomposition. The attention matrix $\mathbf{A}$ provides bimodal explainability.
\textit{Tensor dimension flow:} $x_f \in \mathbb{R}^{3 \times 224 \times 224} \rightarrow f_f \in \mathbb{R}^{1408 \times 7 \times 7} \rightarrow z_f \in \mathbb{R}^{256}$; $x_p \in \mathbb{R}^{3 \times 224 \times 224} \rightarrow f_p \in \mathbb{R}^{512 \times 7 \times 7} \rightarrow z_p \in \mathbb{R}^{256}$; combined sequence $\in \mathbb{R}^{98 \times 256}$ after projection and flattening.}
\label{fig:architecture}
\end{figure*}

The dimension flow through the network is designed to preserve spatial structure until the fusion stage. Face features retain 1408 channels (EfficientNet-B2's output dimension), while fingerprint features use 512 channels. Both are spatially downsampled to $7 \times 7$, yielding 49 tokens per modality. After flattening and projection, the transformer operates on a sequence of $98 \times 256$ tokens, enabling fine-grained token-level cross-attention.

\subsection{Modality-Specific Encoders}
\label{sec:encoders}

\subsubsection{Face Encoder}

The face encoder $\mathcal{E}_f$ is an EfficientNet-B2 \cite{tan2019efficientnet} initialized from ImageNet-pretrained weights via the timm library \cite{wightman2019timm}. EfficientNet-B2 offers a favorable accuracy-efficiency trade-off: with 9.2M parameters and 1.0B FLOPs, it outperforms ResNet-50 (25.6M parameters) on ImageNet while requiring fewer resources. The encoder processes $x_f \in \mathbb{R}^{B \times 3 \times 224 \times 224}$ through a compound scaling of depth, width, and resolution to produce a feature map:
\begin{equation}
\label{eq:face_encoder}
f_f = \mathcal{E}_f(x_f) \in \mathbb{R}^{B \times 1408 \times 7 \times 7},
\end{equation}
where $B$ denotes the batch size. The $7 \times 7$ spatial resolution provides 49 face tokens for subsequent cross-attention, balancing spatial granularity with computational cost. We select EfficientNet over ViT alternatives (e.g., TransFace \cite{dan2023transface}, LVFace \cite{you2025lvface}) because CNN-based encoders generalize better under limited training data and produce better-calibrated spatial feature maps for Grad-CAM explainability (Section \ref{sec:explainability}).

\subsubsection{Fingerprint Encoder}

The fingerprint encoder $\mathcal{E}_p$ is a custom ResNet-style CNN with four residual blocks \cite{he2016deep}, specifically designed for ridge-pattern extraction. Unlike face recognition, where ImageNet pretraining provides strong initialization, fingerprint data has fundamentally different texture statistics (oriented ridge patterns, frequency-domain structure). A custom architecture avoids the domain gap inherent in repurposing natural-image encoders. The encoder produces:
\begin{equation}
\label{eq:fingerprint_encoder}
f_p = \mathcal{E}_p(x_p) \in \mathbb{R}^{B \times 512 \times 7 \times 7},
\end{equation}
matching the spatial resolution of the face encoder for token alignment. Each residual block uses $3 \times 3$ convolutions with batch normalization \cite{ioffe2015batch} and ReLU activation, with channel dimensions [64, 128, 256, 512]. Strided convolutions at blocks 2 and 4 reduce spatial resolution from $224 \times 224$ to $7 \times 7$. The total parameter count is 4.2M, an order of magnitude smaller than the face encoder, reflecting the more constrained visual structure of fingerprint patterns.

\subsubsection{Quality Estimator}

A lightweight quality estimator $\mathcal{Q}_i$ accompanies each encoder to produce a scalar quality score $q_i \in [0, 1]$. The estimator is a 4-block CNN with global average pooling followed by a sigmoid-activated fully-connected layer. For face quality, we optionally initialize from a pretrained SER-FIQ \cite{terhorst2020serfiq} model; for fingerprint quality, the estimator is trained from scratch to predict a proxy for NFIQ 2.0 scores \cite{tabassi2021nist}. The quality estimator adds only 0.3M parameters per modality. The output quality score serves two purposes: it modulates cross-attention weights (Section \ref{sec:qmcatt}) and contributes to the uncertainty budget (Section \ref{sec:uncertainty_heads}).

\subsection{Common Projection and Missing-Modality Handling}
\label{sec:projection_missing}

\subsubsection{L2-Normalized Projectors}

Spatial feature maps from each encoder are flattened along the spatial dimensions to produce token sequences. Face features $f_f \in \mathbb{R}^{B \times 1408 \times 7 \times 7}$ are reshaped to $\mathbb{R}^{B \times 49 \times 1408}$; fingerprint features $f_p \in \mathbb{R}^{B \times 512 \times 7 \times 7}$ to $\mathbb{R}^{B \times 49 \times 512}$. Linear projectors $\mathcal{P}_f: \mathbb{R}^{1408} \rightarrow \mathbb{R}^{256}$ and $\mathcal{P}_p: \mathbb{R}^{512} \rightarrow \mathbb{R}^{256}$ map tokens to a common 256-dimensional space, followed by L2 normalization to the unit hypersphere $\mathcal{S}^{255}$:
\begin{equation}
\label{eq:projection}
z_f^{(j)} = \frac{\mathcal{P}_f(f_f^{(j)})}{\|\mathcal{P}_f(f_f^{(j)})\|_2}, \quad z_p^{(j)} = \frac{\mathcal{P}_p(f_p^{(j)})}{\|\mathcal{P}_p(f_p^{(j)})\|_2},
\end{equation}
where $j \in \{1, \dots, 49\}$ indexes the spatial token. L2 normalization is standard in biometric embedding literature \cite{deng2019arcface, kim2022adaface} as it enforces angular discriminability and stabilizes training. The 256-D dimension follows the conventional setting in deep metric learning for biometrics \cite{grosz2024afrnet, qiu2024ifvit}.

\subsubsection{Learnable Missing-Modality Token}

UFM-Transformer introduces a single learnable missing-modality token $\mathbf{t}_{\text{miss}} \in \mathbb{R}^{256}$, shared across both modalities, as a trainable parameter initialized from $\mathcal{N}(0, 0.02^2)$. When modality $i$ is absent ($m_i = 0$), all 49 projected tokens for that modality are replaced with $\mathbf{t}_{\text{miss}}$ plus a learned positional encoding:
\begin{equation}
\label{eq:missing_replacement}
z_i^{(j)} = \begin{cases}
\mathcal{P}_i(f_i^{(j)}) / \|\mathcal{P}_i(f_i^{(j)})\|_2 & \text{if } m_i = 1, \\
\mathbf{t}_{\text{miss}} + \text{PE}_i^{(j)} & \text{if } m_i = 0.
\end{cases}
\end{equation}

The positional encoding $\text{PE}_i^{(j)} \in \mathbb{R}^{256}$ uses sinusoidal functions of token index $j$ following the standard transformer formulation \cite{vaswani2017attention}, disambiguating the spatial origin of each replaced token. The shared $\mathbf{t}_{\text{miss}}$ differs from Cross-Modal Proxy Tokens (CMPT) \cite{reza2025cmpt} in that it is modality-agnostic: a single token serves both face and fingerprint missing cases, reducing parameters and leveraging the observation that the "absence signal" is itself informative. During training, modality dropout (Section \ref{sec:training_strategy}) ensures the token learns to encode "available modality should compensate."

\subsubsection{Modality Replacement and Sequence Construction}

After replacement, the bimodal token sequence $\mathbf{Z} \in \mathbb{R}^{B \times 98 \times 256}$ is constructed by concatenating face and fingerprint tokens along the sequence dimension. A learnable class token $\mathbf{t}_{\text{cls}} \in \mathbb{R}^{256}$ is prepended, yielding the final transformer input $\mathbf{Z}_{\text{in}} \in \mathbb{R}^{B \times 99 \times 256}$.

\subsection{Quality-Modulated Cross-Attention Transformer}
\label{sec:qmcatt}

The core of UFM-Transformer is the Quality-Modulated Cross-Attention Transformer (QMCAT), which performs bidirectional cross-attention between face and fingerprint tokens with explicit quality-based modulation.

\subsubsection{Two-Stream Cross-Attention}

QMCAT employs a two-stream architecture where face tokens attend to fingerprint keys/values and vice versa. Let $\mathbf{Z}_f = [z_f^{(1)}, \dots, z_f^{(49)}] \in \mathbb{R}^{B \times 49 \times 256}$ and $\mathbf{Z}_p = [z_p^{(1)}, \dots, z_p^{(49)}] \in \mathbb{R}^{B \times 49 \times 256}$ denote the face and fingerprint token sequences. For each attention head $h \in \{1, \dots, H\}$ with head dimension $d_h = D/H = 32$ (where $D=256$, $H=8$), the queries, keys, and values are computed as:
\begin{align}
\mathbf{Q}_f^{(h)} &= \mathbf{Z}_f \mathbf{W}_Q^{(f,h)}, & \mathbf{K}_p^{(h)} &= \mathbf{Z}_p \mathbf{W}_K^{(p,h)}, & \mathbf{V}_p^{(h)} &= \mathbf{Z}_p \mathbf{W}_V^{(p,h)}, \\
\mathbf{Q}_p^{(h)} &= \mathbf{Z}_p \mathbf{W}_Q^{(p,h)}, & \mathbf{K}_f^{(h)} &= \mathbf{Z}_f \mathbf{W}_K^{(f,h)}, & \mathbf{V}_f^{(h)} &= \mathbf{Z}_f \mathbf{W}_V^{(f,h)},
\end{align}
where $\mathbf{W}_Q^{(\cdot)}, \mathbf{W}_K^{(\cdot)}, \mathbf{W}_V^{(\cdot)} \in \mathbb{R}^{256 \times 32}$ are learnable projection matrices.

\subsubsection{Quality Modulation}

The defining innovation of QMCAT is the quality modulation of attention scores. The raw attention logits for the face-querying-fingerprint direction are:
\begin{equation}
\label{eq:attention_logits}
\mathbf{A}_{f \rightarrow p}^{(h)} = \frac{\mathbf{Q}_f^{(h)} (\mathbf{K}_p^{(h)})^{\top}}{\sqrt{d_h}} \in \mathbb{R}^{B \times 49 \times 49},
\end{equation}
before applying softmax, the logits are scaled by the quality product $q_f \cdot q_p$:
\begin{equation}
\label{eq:quality_modulation}
\tilde{\mathbf{A}}_{f \rightarrow p}^{(h)} = \text{softmax}\left( (q_f \cdot q_p) \cdot \mathbf{A}_{f \rightarrow p}^{(h)} \right).
\end{equation}

The quality product $q_f \cdot q_p \in [0, 1]$ acts as a temperature-like scaling factor. When both modalities are high quality ($q_f \cdot q_p \approx 1$), attention weights retain their full discriminative range. When either modality degrades ($q_f \cdot q_p \ll 1$), the scaling flattens the softmax distribution, reducing the influence of potentially noisy cross-modal correspondences. This mechanism places quality modulation at the feature level---the earliest stage where it can influence cross-modal interactions---in contrast to score-level quality fusion which cannot recover lost feature-space information \cite{soleymani2021quality, liang2024deep}.

\subsubsection{Missing-Aware Masking}

When a modality is missing ($m_i = 0$), attention from or to that modality is zeroed via a boolean mask $\mathbf{M} \in \{0, 1\}^{49 \times 49}$:
\begin{equation}
\label{eq:missing_mask}
\tilde{\mathbf{A}}_{f \rightarrow p}^{(h)} = \text{softmax}\left( (q_f \cdot q_p) \cdot \mathbf{A}_{f \rightarrow p}^{(h)} + (1 - \mathbf{M}) \cdot (-\infty) \right),
\end{equation}
where $M_{ij} = 1$ if both tokens $i$ and $j$ correspond to present modalities. This masking ensures that missing-modality tokens (replaced by $\mathbf{t}_{\text{miss}}$) cannot propagate spurious information through the attention mechanism, following the principle that the system should rely exclusively on available evidence.

\subsubsection{Architecture Specification}

QMCAT consists of $L = 4$ transformer layers with $H = 8$ attention heads, 256-D model dimension, and feed-forward dimension 1024 (4$\times$ expansion). Each layer applies, in sequence: (1) quality-modulated cross-attention, (2) residual connection, (3) layer normalization \cite{ba2016layer}, (4) position-wise feed-forward network (FFN: Linear-ReLU-Dropout-Linear), (5) residual connection, and (6) layer normalization. The choice of 4 layers is motivated by AuthFormer \cite{yang2024authformer}, which demonstrated that 2 encoder layers suffice for multimodal biometric authentication; we use 4 layers to provide additional capacity for quality modulation and uncertainty propagation without excessive computation. Dropout rate $p_{\text{drop}} = 0.1$ is applied after attention and FFN sublayers.

The output of QMCAT is the class token representation $\mathbf{t}_{\text{cls}}^{\text{out}} \in \mathbb{R}^{256}$, which aggregates cross-modality information through the self-attention component of each layer, and the full attended token sequence for explainability visualization.

\subsubsection{Forward Pass Algorithm}

Algorithm \ref{alg:forward} provides the complete pseudocode for a single forward pass through QMCAT.

\begin{algorithm}[t]
\caption{UFM-Transformer Forward Pass (QMCAT)}
\label{alg:forward}
\begin{algorithmic}[1]
\Require Face image $x_f$, fingerprint image $x_p$, masks $m_f, m_p$, quality scores $q_f, q_p$
\Ensure Similarity score $\hat{s}$, uncertainty $\sigma_{\text{total}}$
\State $f_f \leftarrow \mathcal{E}_f(x_f) \in \mathbb{R}^{1408 \times 7 \times 7}$
\State $f_p \leftarrow \mathcal{E}_p(x_p) \in \mathbb{R}^{512 \times 7 \times 7}$
\State $z_f \leftarrow \text{L2Norm}(\mathcal{P}_f(\text{Flatten}(f_f))) \in \mathbb{R}^{49 \times 256}$
\State $z_p \leftarrow \text{L2Norm}(\mathcal{P}_p(\text{Flatten}(f_p))) \in \mathbb{R}^{49 \times 256}$
\If {$m_f = 0$} $z_f \leftarrow \mathbf{t}_{\text{miss}} + \text{PE}_f$ \textbf{for all} 49 tokens \EndIf
\If {$m_p = 0$} $z_p \leftarrow \mathbf{t}_{\text{miss}} + \text{PE}_p$ \textbf{for all} 49 tokens \EndIf
\State $\mathbf{Z}_{\text{in}} \leftarrow \text{Concat}([\mathbf{t}_{\text{cls}}; z_f; z_p]) \in \mathbb{R}^{99 \times 256}$
\For {$l = 1$ \textbf{to} $L$}
    \State // Quality-modulated cross-attention
    \State $\mathbf{A}_{f \rightarrow p} \leftarrow \frac{\mathbf{Q}_f \mathbf{K}_p^{\top}}{\sqrt{d_h}}$; $\quad \mathbf{A}_{p \rightarrow f} \leftarrow \frac{\mathbf{Q}_p \mathbf{K}_f^{\top}}{\sqrt{d_h}}$
    \State $\tilde{\mathbf{A}}_{f \rightarrow p} \leftarrow \text{softmax}\big((q_f \cdot q_p) \cdot \mathbf{A}_{f \rightarrow p} + (1-\mathbf{M}) \cdot (-\infty)\big)$
    \State $\tilde{\mathbf{A}}_{p \rightarrow f} \leftarrow \text{softmax}\big((q_f \cdot q_p) \cdot \mathbf{A}_{p \rightarrow f} + (1-\mathbf{M}) \cdot (-\infty)\big)$
    \State $\mathbf{O}_f \leftarrow \tilde{\mathbf{A}}_{f \rightarrow p} \mathbf{V}_p$; $\quad \mathbf{O}_p \leftarrow \tilde{\mathbf{A}}_{p \rightarrow f} \mathbf{V}_f$
    \State $\mathbf{Z}_{\text{out}} \leftarrow \text{LayerNorm}(\mathbf{Z}_{\text{in}} + \text{Concat}([\mathbf{0}; \mathbf{O}_f; \mathbf{O}_p]))$
    \State $\mathbf{Z}_{\text{out}} \leftarrow \text{LayerNorm}(\mathbf{Z}_{\text{out}} + \text{FFN}(\mathbf{Z}_{\text{out}}))$
    \State $\mathbf{Z}_{\text{in}} \leftarrow \mathbf{Z}_{\text{out}}$
\EndFor
\State $\mathbf{t}_{\text{cls}}^{\text{out}} \leftarrow \mathbf{Z}_{\text{out}}[0, :]$ // Extract class token
\State // Similarity head (ArcFace margin)
\State $\hat{s} \leftarrow \cos(\mathbf{t}_{\text{cls}}^{\text{out}}, \mathbf{w}_y) - m$ // with additive angular margin $m=0.5$
\State // Uncertainty head (MC Dropout, $T=5$ passes)
\State $\{\hat{s}^{(t)}\}_{t=1}^{T} \leftarrow \text{MCForward}(\mathbf{t}_{\text{cls}}^{\text{out}}, T)$
\State $\sigma_{\text{epistemic}} \leftarrow \text{Var}(\{\hat{s}^{(t)}\})$; $\quad \sigma_{\text{aleatoric}} \leftarrow \text{Mean}(\{\sigma^{(t)}\})$
\State \Return $\hat{s}$, $\sigma_{\text{total}} = \sigma_{\text{epistemic}} + \sigma_{\text{aleatoric}}$
\end{algorithmic}
\end{algorithm}

\subsection{Similarity and Uncertainty Heads}
\label{sec:uncertainty_heads}

\subsubsection{Similarity Head with ArcFace Margin}

The similarity head computes the cosine similarity between the class token $\mathbf{t}_{\text{cls}}^{\text{out}}$ and the target identity prototype $\mathbf{w}_y \in \mathbb{R}^{256}$, with an additive angular margin \cite{deng2019arcface}:
\begin{equation}
\label{eq:arcface}
\mathcal{L}_{\text{arc}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{e^{s \cdot (\cos(\theta_{y_i} + m))}}{e^{s \cdot (\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cdot \cos(\theta_j)}},
\end{equation}
where $\theta_j = \arccos(\mathbf{t}_{\text{cls},i}^{\text{out}} \cdot \mathbf{w}_j)$ is the angle between the class token and prototype $j$, $m = 0.5$ is the additive angular margin, and $s = 30$ is the scaling factor. The margin $m$ enforces an angular separation between classes, improving inter-class discriminability \cite{deng2019arcface, kim2022adaface}. At inference, the similarity between a probe $\mathbf{x}$ and gallery template $\mathbf{x}'$ is:
\begin{equation}
\label{eq:similarity}
\hat{s}(\mathbf{x}, \mathbf{x}') = \frac{(\mathbf{t}_{\text{cls}}^{\text{out}})^{\top} (\mathbf{t}_{\text{cls}}^{\prime\text{out}})}{\|\mathbf{t}_{\text{cls}}^{\text{out}}\|_2 \cdot \|\mathbf{t}_{\text{cls}}^{\prime\text{out}}\|_2}.
\end{equation}

\subsubsection{Uncertainty Head: Monte Carlo Variance Decomposition}

UFM-Transformer estimates uncertainty through Monte Carlo (MC) Dropout \cite{gal2016dropout}, performing $T = 5$ stochastic forward passes with dropout enabled at inference. Let $\{\hat{s}^{(t)}\}_{t=1}^{T}$ denote the similarity scores from $T$ passes. Following the Bayesian deep learning framework \cite{kendall2017uncertainties}, the total variance decomposes into epistemic (model) and aleatoric (data) components:
\begin{align}
\label{eq:variance_decomposition}
\sigma_{\text{epistemic}}^2 &= \frac{1}{T} \sum_{t=1}^{T} \left(\hat{s}^{(t)} - \bar{s}\right)^2, \\
\sigma_{\text{aleatoric}}^2 &= \frac{1}{T} \sum_{t=1}^{T} \sigma^{(t)2}, \\
\sigma_{\text{total}}^2 &= \sigma_{\text{epistemic}}^2 + \sigma_{\text{aleatoric}}^2,
\end{align}
where $\bar{s} = \frac{1}{T}\sum_{t=1}^{T} \hat{s}^{(t)}$ and $\sigma^{(t)}$ is the predicted log-variance from the $t$-th pass. The epistemic component captures model uncertainty (disagreement between forward passes due to parameter uncertainty), while the aleatoric component captures inherent data noise \cite{kendall2017uncertainties}. The choice of $T = 5$ balances computational cost with variance estimation accuracy; $T \geq 5$ yields diminishing returns in practice \cite{gal2016dropout}.

\subsubsection{Confidence Interval and Rejection Option}

The final verification output is a confidence interval (Eq. \ref{eq:verification_objective}) with $\kappa = 2$ (approximating 95\% confidence under Gaussianity). When the interval $[\hat{s} - \kappa\sigma_{\text{total}}, \hat{s} + \kappa\sigma_{\text{total}}]$ spans the decision threshold $\tau$, the system rejects the decision:
\begin{equation}
\label{eq:rejection}
\text{decision} = \begin{cases}
\text{genuine} & \text{if } \hat{s} - \kappa\sigma_{\text{total}} > \tau, \\
\text{impostor} & \text{if } \hat{s} + \kappa\sigma_{\text{total}} < \tau, \\
\text{uncertain} & \text{otherwise (rejection)}.
\end{cases}
\end{equation}
This formulation provides principled abstention under uncertainty, following the conformal prediction framework \cite{balomenos2018automatic} and connecting to the rejection option literature \cite{chow1970optimum}.

\subsection{Bimodal Explainability}
\label{sec:explainability}

\subsubsection{Cross-Attention Visualization}

The cross-attention weight matrices $\tilde{\mathbf{A}}_{f \rightarrow p} \in \mathbb{R}^{49 \times 49}$ and $\tilde{\mathbf{A}}_{p \rightarrow f} \in \mathbb{R}^{49 \times 49}$ provide inherent explainability by revealing which face regions attend to which fingerprint regions for identity matching. Each matrix row corresponds to a query token (one spatial location in the source modality) and each column to a key token (one spatial location in the target modality). By reshaping attention rows back to $7 \times 7$ and upsampling to the original $224 \times 224$ resolution, we obtain spatial attention heatmaps for both modalities.

\subsubsection{Grad-CAM Bimodal Explanations}

Complementing attention visualization, gradient-weighted class activation mapping (Grad-CAM) \cite{selvaraju2017gradcam} is applied to both encoders to highlight regions that most influence the similarity score. For the face encoder, gradients $\partial \hat{s} / \partial f_f$ flow backward from the similarity head to produce a class-discriminative localization map. For the fingerprint encoder, gradients $\partial \hat{s} / \partial f_p$ reveal ridge patterns and minutiae regions that contribute to matching. The combined bimodal explanation overlays both Grad-CAM heatmaps weighted by their modality quality scores.

\subsubsection{Attention Heatmaps for Failure Case Analysis}

Cross-attention heatmaps serve a diagnostic function: in genuine pairs where attention is diffuse (low maximum attention weight), the quality scores typically indicate degraded inputs. In impostor pairs where attention concentrates on spurious correspondences (e.g., a dark face region aligning with a noisy fingerprint valley), the uncertainty head outputs elevated $\sigma_{\text{total}}$, flagging the decision as unreliable. This dual use of attention weights for both fusion and explanation follows the principle that architectural components should serve multiple purposes simultaneously \cite{gazi2026minutiae, lu2024towards}.

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{attention_mechanism_detail.pdf}
\caption{\textbf{Quality-Modulated Cross-Attention mechanism (detail).} Face tokens $\mathbf{Z}_f$ and fingerprint tokens $\mathbf{Z}_p$ form query-key-value pairs for bidirectional cross-attention. Raw attention logits $\mathbf{A}$ are scaled by the quality product $q_f \cdot q_p$ before softmax; missing modalities are masked with $-\infty$. The attended outputs $\mathbf{O}_f, \mathbf{O}_p$ are concatenated, projected, and added as a residual. LayerNorm and FFN follow (not shown).}
\label{fig:attention_detail}
\end{figure}

\subsection{Training Strategy}
\label{sec:training_strategy}

\subsubsection{Composite Loss Function}

UFM-Transformer is trained with a three-term composite loss:
\begin{equation}
\label{eq:composite_loss}
\mathcal{L} = \mathcal{L}_{\text{triplet}} + \lambda_1 \mathcal{L}_{\text{arcface}} + \lambda_2 \mathcal{L}_{\text{uncertainty}},
\end{equation}
where $\mathcal{L}_{\text{triplet}}$ is the batch-hard triplet loss \cite{hermans2017defense} with margin $\alpha = 0.5$, $\mathcal{L}_{\text{arcface}}$ is the additive angular margin loss (Eq. \ref{eq:arcface}), and $\mathcal{L}_{\text{uncertainty}}$ penalizes high uncertainty on correctly classified samples while allowing uncertainty on errors:
\begin{equation}
\label{eq:uncertainty_loss}
\mathcal{L}_{\text{uncertainty}} = \frac{1}{B} \sum_{i=1}^{B} \mathbb{1}[\hat{y}_i = y_i] \cdot \sigma_{\text{total},i}^2 + \mathbb{1}[\hat{y}_i \neq y_i] \cdot \max(0, \beta - \sigma_{\text{total},i})^2,
\end{equation}
where $\beta = 1.0$ is a target uncertainty for misclassified samples. This loss encourages the model to be confident when correct and uncertain when wrong, following the data uncertainty learning framework \cite{chang2020data}. We set $\lambda_1 = 1.0$ and $\lambda_2 = 0.1$ based on initial validation experiments.

\subsubsection{Two-Phase Training Protocol}

Training proceeds in two phases to ensure stable convergence:

\textbf{Phase 1 -- Unimodal Pretraining (50 epochs).} The encoders $\mathcal{E}_f$, $\mathcal{E}_p$ and projectors $\mathcal{P}_f$, $\mathcal{P}_p$ are trained independently with $\mathcal{L}_{\text{arcface}}$ on unimodal batches. Face and fingerprint networks do not interact. This phase ensures each encoder learns a strong unimodal representation before the more complex cross-attention mechanism is introduced \cite{dan2023transface, grosz2024afrnet}.

\textbf{Phase 2 -- Joint Fine-Tuning (100 epochs).} The full model is trained end-to-end with the composite loss (Eq. \ref{eq:composite_loss}). During this phase, modality dropout \cite{neverova2016moddrop} is applied: with probability $p_{\text{drop}} = 0.3$, one randomly selected modality is masked ($m_i = 0$) for the entire batch. This forces the model to learn robust cross-modal compensation. The dropout rate of 30\% follows the ModDrop literature \cite{neverova2016moddrop} and balances robustness with performance on full bimodal inputs.

\subsubsection{Optimization Hyperparameters}

Optimization uses AdamW \cite{loshchilov2018decoupled} with weight decay $10^{-4}$, initial learning rate $10^{-4}$, and cosine annealing schedule \cite{loshchilov2016sgdr} with warm restart every 10 epochs. Gradient clipping at norm 1.0 prevents instability during the initial cross-attention warm-up. Training batches contain 64 bimodal samples (32 identities $\times$ 2 samples). All experiments use PyTorch 2.0 with mixed-precision training on NVIDIA A100 GPUs.

\subsubsection{Quality Estimator Training}

Quality estimators are trained in parallel with Phase 1 using a binary cross-entropy loss against pseudo-labels: face quality is derived from the confidence of a pretrained face recognition model (lower confidence $\rightarrow$ lower quality), while fingerprint quality uses NFIQ 2.0 scores mapped to $[0, 1]$. This weakly-supervised approach avoids manual quality annotation \cite{soleymani2021quality}.

\end{document}
