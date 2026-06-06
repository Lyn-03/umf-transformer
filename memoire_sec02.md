\chapter{Le Mod\`{e}le UFM-Transformer}
\label{chap:modele}

Ce chapitre pr\'{e}sente en d\'{e}tail l'architecture UFM-Transformer (\emph{Uncertainty-aware Fusion Multimodal Transformer}), con\c{c}ue pour la v\'{e}rification biom\'{e}trique multimodale associant visage et empreinte digitale. L'expos\'{e} vise une reproductibilit\'{e} int\'{e}grale : chaque choix de conception est justifi\'{e}, chaque hyperparam\`{e}tre est explicit\'{e}, et les flux de dimensions tensorielles sont indiqu\'{e}s \`{a} chaque \'{e}tape du traitement. L'architecture unifie quatre fonctions rarement conjugu\'{e}es dans la litt\'{e}rature biom\'{e}trique : la fusion par attention crois\'{e}e entre modalit\'{e}s, la gestion native des modalit\'{e}s manquantes, la quantification de l'incertitude, et l'explicabilit\'{e} bimodale inh\'{e}rente \`{a} l'attention.

\section{Formulation du Probl\`{e}me}
\label{sec:formulation}

\subsection{Notations et d\'{e}finitions formelles}

On d\'{e}finit l'espace des entr\'{e}es bimodales. Une image faciale est not\'{e}e $\mathbf{x}_f \in \mathcal{X}_f = \mathbb{R}^{3 \times H_f \times W_f}$, o\`{u} $H_f = W_f = 224$ d\'{e}signent la r\'{e}solution spatiale et le canal de profondeur $3$ correspond aux composantes RGB. Une image d'empreinte digitale est not\'{e}e $\mathbf{x}_p \in \mathcal{X}_p = \mathbb{R}^{1 \times H_p \times W_p}$, avec $H_p = W_p = 224$, le canal unique correspondant \`{a} l'intensit\'{e} en niveaux de gris. Un \'{e}chantillon d'entra\^{i}nement est un triplet $(\mathbf{x}_f, \mathbf{x}_p, y)$, o\`{u} $y \in \{0, 1\}$ est l'\'{e}tiquette binaire indiquant si la paire appartient \`{a} la m\^{e}me identit\'{e} ($y=1$) ou \`{a} des identit\'{e}s diff\'{e}rentes ($y=0$).

L'ensemble des identit\'{e}s est not\'{e} $\mathcal{I} = \{1, \ldots, N_{\text{id}}\}$. Chaque identit\'{e} $i$ est repr\'{e}sent\'{e}e dans la galerie par un gabarit (\emph{template}) bimodal $\mathbf{G}_i = (\mathbf{g}_f^{(i)}, \mathbf{g}_p^{(i)})$, o\`{u} $\mathbf{g}_f^{(i)}$ et $\mathbf{g}_p^{(i)}$ d\'{e}signent respectivement les caract\'{e}ristiques faciales et papillaires de r\'{e}f\'{e}rence. \`{A} l'inference, une requ\^{e}te $\mathbf{Q} = (\mathbf{q}_f, \mathbf{q}_p)$ est compar\'{e}e \`{a} l'ensemble des gabarits de la galerie.

On introduit le vecteur de pr\'{e}sence modalitaire $\mathbf{m} = (m_f, m_p) \in \{0, 1\}^2$, o\`{u} $m_f = 1$ (resp. $m_p = 1$) indique que la modalit\'{e} visage (resp. empreinte) est disponible. Le cas $\mathbf{m} = (1, 1)$ correspond au sc\'{e}nario bimodal complet ; les cas $(1, 0)$ et $(0, 1)$ d\'{e}crivent les d\'{e}faillances unimodales de capteur.

\subsection{Objectif de v\'{e}rification ouverte avec qualit\'{e} variable et modalit\'{e}s partielles}

Le probl\`{e}me de v\'{e}rification biom\'{e}trique ouverte (\emph{open-set verification}) consiste \`{a} apprendre une fonction de similarit\'{e} param\'{e}tr\'{e}e $s_\theta : \mathcal{X}_f \times \mathcal{X}_p \times \mathcal{X}_f \times \mathcal{X}_p \rightarrow \mathbb{R}$ telle que :
\begin{equation}
\label{eq:verification}
s_\theta(\mathbf{q}_f, \mathbf{q}_p; \mathbf{g}_f^{(i)}, \mathbf{g}_p^{(i)}) > \tau \quad \Leftrightarrow \quad \mathbf{Q} \text{ et } \mathbf{G}_i \text{ partagent la m\^{e}me identit\'{e}},
\end{equation}
o\`{u} $\tau \in \mathbb{R}$ est un seuil de d\'{e}cision. Dans le cadre de ce m\'{e}moire, trois contraintes suppl\'{e}mentaires \'{e}tendent ce probl\`{e}me standard.

\textbf{Contrainte de qualit\'{e} variable.} Les scores de qualit\'{e} $q_f, q_p \in [0, 1]$ sont associ\'{e}s \`{a} chaque image. La fonction de similarit\'{e} doit \^{e}tre modul\'{e}e par ces scores de mani\`{e}re \`{a} att\'{e}nuer l'influence des modalit\'{e}s d\'{e}grad\'{e}es. Formellement, on impose que la similarit\'{e} $s$ d\'{e}pende de $q_f$ et $q_p$ au premier ordre : $\frac{\partial s}{\partial q_f} > 0$ et $\frac{\partial s}{\partial q_p} > 0$ en moyenne sur les paires g\'{e}nuines.

\textbf{Contrainte de modalit\'{e}s partielles.} Lorsqu'une modalit\'{e} est absente ($m_k = 0$), le syst\`{e}me doit produire une similarit\'{e} d\'{e}grad\'{e}e gracieusement, sans n\'{e}cessiter de branche architecturale sp\'{e}cifique. On formalise cette exigence par la condition de continuit\'{e} :
\begin{equation}
\label{eq:continuite}
\lim_{q_k \rightarrow 0} s_\theta(\cdot; q_f, q_p) = s_\theta^{\setminus k}(\cdot),
\end{equation}
o\`{u} $s_\theta^{\setminus k}$ d\'{e}signe la similarit\'{e} unimodale obtenue lorsque seule la modalit\'{e} compl\'{e}mentaire est disponible.

\textbf{Contrainte d'incertitude calibr\'{e}e.} Le syst\`{e}me doit produire, en plus du score de similarit\'{e} $s$, une estimation d'incertitude $\sigma^2 \in \mathbb{R}_+$ telle que la probabilit\'{e} de bonne classification conditionnellement \`{a} $\sigma^2$ satisfasse :
\begin{equation}
\label{eq:calibration}
\mathbb{P}(\hat{y} = y \mid \sigma^2 = v) \approx 1 - v/\sigma^2_{\max},
\end{equation}
c'est-\`{a}-dire que l'incertitude doit \^{e}tre une pr\'{e}diction calibr\'{e}e de la probabilit\'{e} d'erreur.

\section{Vue d'Ensemble de l'Architecture}
\label{sec:vue-ensemble}

\subsection{Pipeline global : encodeurs $\rightarrow$ projecteurs $\rightarrow$ transformer cross-modal $\rightarrow$ t\^{e}tes de d\'{e}cision}

L'architecture UFM-Transformer s'organise en quatre modules fonctionnels connect\'{e}s en s\'{e}rie, chacun r\'{e}solvant un sous-probl\`{e}me distinct.

Le \textbf{module d'encodage modalitaire} (\S\ref{sec:encodeurs}) extrait des repr\'{e}sentations sp\'{e}cialis\'{e}es pour chaque modalit\'{e}. L'encodeur visage $\mathcal{E}_f$ est un EfficientNet-B2 \cite{tan2019efficientnet} pr\'{e}-entra\^{i}n\'{e} sur ImageNet, produisant une carte de caract\'{e}ristiques $\mathbf{z}_f \in \mathbb{R}^{1408 \times 7 \times 7}$. L'encodeur empreinte $\mathcal{E}_p$ est un CNN r\'{e}siduel personnalis\'{e}, produisant $\mathbf{z}_p \in \mathbb{R}^{512 \times 7 \times 7}$. Deux estimateurs de qualit\'{e} l\'{e}gers $\mathcal{Q}_f$ et $\mathcal{Q}_p$, impl\'{e}ment\'{e}s comme des CNN \`{a} trois couches, produisent les scores $q_f, q_p \in [0, 1]$.

Le \textbf{module de projection} (\S\ref{sec:projecteurs}) aplait et projette les caract\'{e}ristiques sur une hypersph\`{e}re commune de dimension $d = 256$. Les projecteurs $\mathcal{P}_f : \mathbb{R}^{1408 \times 7 \times 7} \rightarrow \mathbb{S}^{255}$ et $\mathcal{P}_p : \mathbb{R}^{512 \times 7 \times 7} \rightarrow \mathbb{S}^{255}$ produisent respectivement $\mathbf{h}_f$ et $\mathbf{h}_p$, avec $\|\mathbf{h}_f\|_2 = \|\mathbf{h}_p\|_2 = 1$. Un token apprenable $\mathbf{t}_{\text{miss}} \in \mathbb{R}^{256}$ remplace la repr\'{e}sentation d'une modalit\'{e} absente.

Le \textbf{module de fusion par attention crois\'{e}e} (\S\ref{sec:attention}) constitue le c\oe ur architectural. Un empilement de $L = 4$ couches de transformer \`{a} attention crois\'{e}e bidirectionnelle, chacune avec $H = 8$ t\^{e}tes et une dimension latente $d = 256$, produit un token fusionn\'{e} $\mathbf{c} \in \mathbb{R}^{256}$. Les scores de qualit\'{e} $q_f$ et $q_p$ modulent les logits d'attention avant le \emph{softmax}, garantissant que les modalit\'{e}s d\'{e}grad\'{e}es contribuent moins au r\'{e}sultat fusionn\'{e}.

Le \textbf{module de d\'{e}cision} (\S\ref{sec:similarite}) comprend deux t\^{e}tes. La t\^{e}te de similarit\'{e} $\mathcal{S}$ calcule un score cosinus avec marge additive ArcFace \cite{deng2019arcface} de param\`{e}tres $m = 0{,}5$ et $s = 30$. La t\^{e}te d'incertitude $\mathcal{U}$ effectue $K = 5$ passes avant avec dropout actif \cite{gal2016dropout} pour produire une d\'{e}composition de l'incertitude en composantes al\'{e}atoire et \'{e}pist\'{e}mique, ainsi qu'un intervalle de confiance associ\'{e} \`{a} la d\'{e}cision.

La figure \ref{fig:architecture} synth\'{e}tise l'ensemble du pipeline. Le flux de donn\'{e}es y est repr\'{e}sent\'{e} avec les dimensions tensorielles \`{a} chaque \'{e}tape.

\subsection{Figure : diagramme d'architecture complet avec flux de donn\'{e}es}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.95\textwidth]{architecture_ufm.pdf}
\caption{Architecture compl\`{e}te de l'UFM-Transformer. Les dimensions tensorielles sont indiqu\'{e}es \`{a} chaque \'{e}tape. Les fl\`{e}ches en trait plein repr\'{e}sentent les flux de donn\'{e}es ; les fl\`{e}ches en pointill\'{e}s repr\'{e}sentent les scores de qualit\'{e} modulant l'attention. Le symbole $\otimes$ d\'{e}note la modulation multiplicative des logits d'attention par le produit $q_f \cdot q_p$.}
\label{fig:architecture}
\end{figure}

\section{Encodeurs Sp\'{e}cifiques par Modalit\'{e}}
\label{sec:encodeurs}

\subsection{Encodeur visage : EfficientNet-B2, pr\'{e}-entra\^{i}nement ImageNet, extraction de cartes de caract\'{e}ristiques}

L'encodeur visage adopte EfficientNet-B2 \cite{tan2019efficientnet}, dont le profil de complexit\'{e} (9,2 M de param\`{e}tres, 0,69 GFLOPs \`{a} la r\'{e}solution $224 \times 224$) offre un compromis favorable entre capacit\'{e} de repr\'{e}sentation et co\^{u}t de calcul. Ce r\'{e}seau, pr\'{e}-entra\^{i}n\'{e} sur ImageNet-1K (1000 classes, 1,28 M d'images), est gel\'{e} lors des 50 premi\`{e}res \'{e}poques de pr\'{e}-entra\^{i}nement unimodal, puis d\'{e}gel\'{e} avec un taux d'apprentissage r\'{e}duit de $\eta = 10^{-5}$ lors du \emph{fine-tuning} joint.

Une entr\'{e}e faciale $\mathbf{x}_f \in \mathbb{R}^{3 \times 224 \times 224}$, pr\'{e}alablement align\'{e}e par d\'{e}tection des points caract\'{e}ristiques et normalis\'{e}e par les moyennes et \'{e}carts-types d'ImageNet ($\mu = [0{,}485, 0{,}456, 0{,}406]$, $\sigma = [0{,}229, 0{,}224, 0{,}225]$), traverse les sept blocs de convolution compos\'{e}s du r\'{e}seau. La sortie du cinqui\`{e}me bloc, avant le \emph{global average pooling}, constitue la carte de caract\'{e}ristiques retenue :
\begin{equation}
\mathbf{z}_f = \mathcal{E}_f(\mathbf{x}_f) \in \mathbb{R}^{1408 \times 7 \times 7},
\end{equation}
o\`{u} $1408$ est la profondeur du canal de sortie de l'avant-dernier bloc et $7 \times 7$ la r\'{e}solution spatiale r\'{e}siduelle. Ce choix de capturer les caract\'{e}ristiques avant agr\'{e}gation spatiale pr\'{e}serve l'information de localisation n\'{e}cessaire \`{a} l'attention crois\'{e}e ult\'{e}rieure : chaque position spatiale $(i, j)$ de $\mathbf{z}_f$ peut \^{e}tre mise en correspondance avec une r\'{e}gion de l'image faciale originale via le facteur de sous-\'{e}chantillonnage total $32 \times$.

\subsection{Encodeur empreinte : CNN r\'{e}siduel personnalis\'{e} pour les motifs de cr\^{e}tes}

Les empreintes digitales poss\`{e}dent des propri\'{e}t\'{e}s structurelles distinctes : invariance par translation, orientation locale des cr\^{e}tes, et richesse en minuties. Un encodeur d\'{e}di\'{e}, plut\^{o}t qu'un r\'{e}seau de vision g\'{e}n\'{e}rique, permet d'exploiter ces sp\'{e}cificit\'{e}s. L'encodeur $\mathcal{E}_p$ est un CNN r\'{e}siduel \`{a} 18 couches, inspir\'{e} de l'architecture de Grosz \& Jain \cite{grosz2024afrnet} pour la reconnaissance d'empreintes.

La structure de $\mathcal{E}_p$ se compose de quatre blocs r\'{e}siduels. Chaque bloc contient deux couches de convolution $3 \times 3$ avec normalisation par batch et activation ReLU, suivies d'une connexion r\'{e}siduelle. Le nombre de filtres progresse de 64 (bloc 1) \`{a} 128 (bloc 2), 256 (bloc 3) et 512 (bloc 4). Un max-pooling $2 \times 2$ avec stride 2 est appliqu\'{e} entre chaque bloc. La sortie de l'avant-dernier bloc, avant agr\'{e}gation spatiale, est :
\begin{equation}
\mathbf{z}_p = \mathcal{E}_p(\mathbf{x}_p) \in \mathbb{R}^{512 \times 7 \times 7},
\end{equation}
o\`{u} $\mathbf{x}_p \in \mathbb{R}^{1 \times 224 \times 224}$ est l'image d'empreinte, pr\'{e}alablement normalis\'{e}e \`{a} l'intervalle $[0, 1]$ et augment\'{e}e par rotation al\'{e}atoire dans $[-15^{\circ}, +15^{\circ}]$ ainsi que par ajustement de contraste.

Le nombre total de param\`{e}tres de $\mathcal{E}_p$ est de 11,2 M. Cet encodeur est entra\^{i}n\'{e} from scratch (sans pr\'{e}-entra\^{i}nement) avec la perte ArcFace lors de la phase 1, ce qui s'av\`{e}re suffisant compte tenu de la taille du corpus d'empreintes et de la sp\'{e}cificit\'{e} du domaine.

\subsection{Estimateur de qualit\'{e} : CNN l\'{e}ger par modalit\'{e}, score dans $[0,1]$}

Chaque modalit\'{e} est munie d'un estimateur de qualit\'{e} ind\'{e}pendant, impl\'{e}ment\'{e} comme un r\'{e}seau l\'{e}ger \`{a} trois couches de convolution. Ces r\'{e}seaux sont motiv\'{e}s par l'observation, document\'{e}e par Soleymani \emph{et al.} \cite{soleymani2021quality}, selon laquelle la pond\'{e}ration explicite par la qualit\'{e} \`{a} l'\'{e}tape de fusion am\'{e}liore le taux d'acceptation de plus de 30\% \`{a} FAR $= 10^{-4}$.

Pour le visage, l'estimateur $\mathcal{Q}_f$ prend l'image $\mathbf{x}_f$ en entr\'{e}e et produit un scalaire $q_f \in [0, 1]$ par le pipeline suivant :
\begin{align}
\mathbf{u}_1 &= \text{ReLU}\bigl(\text{BN}(\text{Conv}_{3 \times 3}^{32}(\mathbf{x}_f))\bigr) &&\in \mathbb{R}^{32 \times 112 \times 112}, \\
\mathbf{u}_2 &= \text{ReLU}\bigl(\text{BN}(\text{Conv}_{3 \times 3}^{64}(\mathbf{u}_1))\bigr) &&\in \mathbb{R}^{64 \times 56 \times 56}, \\
\mathbf{u}_3 &= \text{ReLU}\bigl(\text{BN}(\text{Conv}_{3 \times 3}^{128}(\mathbf{u}_2))\bigr) &&\in \mathbb{R}^{128 \times 28 \times 28}, \\
q_f &= \sigma\bigl(\text{FC}_{512 \rightarrow 1}(\text{GAP}(\mathbf{u}_3))\bigr),
\end{align}
o\`{u} GAP d\'{e}signe le \emph{global average pooling}, FC une couche enti\`{e}rement connect\'{e}e, et $\sigma$ la fonction sigmo\'{i}de. L'estimateur $\mathcal{Q}_p$ pour les empreintes partage la m\^{e}me structure mais re\c{c}oit un canal unique en entr\'{e}e. Chaque estimateur compte 0,18 M de param\`{e}tres. Les scores $q_f$ et $q_p$ sont supervis\'{e}s indirectement : ils ne requi\`{e}rent pas d'annotations de qualit\'{e} explicites mais sont appris de mani\`{e}re faiblement supervis\'{e}e via la perte composite (\S\ref{sec:perte}), suivant le principe de DUL \cite{chang2020data} selon lequel l'incertitude apprise corr\`{e}le naturellement avec la d\'{e}gradation de la qualit\'{e}.

\section{Projecteur Commun et Gestion des Modalit\'{e}s Manquantes}
\label{sec:projecteurs}

\subsection{Projecteurs denses avec normalisation L2 : projection sur l'hypersph\`{e}re unit\'{e}}

Les repr\'{e}sentations produites par les encodeurs, $\mathbf{z}_f \in \mathbb{R}^{1408 \times 7 \times 7}$ et $\mathbf{z}_p \in \mathbb{R}^{512 \times 7 \times 7}$, diff\`{e}rent \`{a} la fois par leur profondeur de canal et leur s\'{e}mantique interne. Un projecteur commun est requis pour les ramener \`{a} un espace partag\'{e} o\`{u} l'attention crois\'{e}e peut op\'{e}rer de mani\`{e}re sym\'{e}trique.

Le projecteur visage $\mathcal{P}_f$ est une s\'{e}quence de couches enti\`{e}rement connect\'{e}es suivie d'une normalisation L2 :
\begin{equation}
\label{eq:projecteur-face}
\mathbf{h}_f = \frac{\text{ReLU}(\mathbf{W}_{f,2} \cdot \text{ReLU}(\mathbf{W}_{f,1} \cdot \text{Flatten}(\mathbf{z}_f) + \mathbf{b}_{f,1}) + \mathbf{b}_{f,2})}{\bigl\| \text{ReLU}(\cdots) \bigr\|_2},
\end{equation}
o\`{u} $\mathbf{W}_{f,1} \in \mathbb{R}^{512 \times (1408 \cdot 49)}$, $\mathbf{W}_{f,2} \in \mathbb{R}^{256 \times 512}$, et $\mathbf{h}_f \in \mathbb{S}^{255} \subset \mathbb{R}^{256}$ v\'{e}rifie $\|\mathbf{h}_f\|_2 = 1$. Le \emph{flattening} de $\mathbf{z}_f$ produit un vecteur de dimension $1408 \times 7 \times 7 = 69\,032$.

Le projecteur empreinte $\mathcal{P}_p$ suit la m\^{e}me architecture avec des poids distincts :
\begin{equation}
\mathbf{h}_p = \frac{\text{ReLU}(\mathbf{W}_{p,2} \cdot \text{ReLU}(\mathbf{W}_{p,1} \cdot \text{Flatten}(\mathbf{z}_p) + \mathbf{b}_{p,1}) + \mathbf{b}_{p,2})}{\bigl\| \text{ReLU}(\cdots) \bigr\|_2},
\end{equation}
o\`{u} $\mathbf{W}_{p,1} \in \mathbb{R}^{512 \times (512 \cdot 49)}$, $\mathbf{W}_{p,2} \in \mathbb{R}^{256 \times 512}$, et $\mathbf{h}_p \in \mathbb{S}^{255}$. La dimension $d = 256$ de l'espace latent commun est choisie comme compromis entre capacit\'{e} de repr\'{e}sentation et complexit\'{e} computationnelle du transformer ; elle correspond \`{a} la dimension utilis\'{e}e par les architectures biom\'{e}triques multimodales de l'\'{e}tat de l'art \cite{talreja2021deephashing, yang2024authformer}.

\subsection{Token apprenable de remplacement : $\mathbf{t}_{\text{miss}} \in \mathbb{R}^{256}$ param\`{e}tre entra\^{i}nable}

La gestion des modalit\'{e}s manquantes constitue un des apports fondamentaux de l'UFM-Transformer. Contrairement aux approches par \emph{fallback} au niveau score \cite{elbousty2025}, qui n\'{e}cessitent des branches architecturales distinctes, notre m\'{e}thode op\`{e}re au niveau du token.

On introduit un vecteur apprenable $\mathbf{t}_{\text{miss}} \in \mathbb{R}^{256}$, initialis\'{e} selon une loi normale $\mathcal{N}(\mathbf{0}, 0{,}02^2 \cdot \mathbf{I})$. Ce token joue le r\^{o}le de proxy pour la modalit\'{e} absente, en s'inspirant des \emph{Cross-Modal Proxy Tokens} (CMPT) propos\'{e}s par Reza \emph{et al.} \cite{reza2025cmpt}. Lorsqu'une modalit\'{e} est manquante, $\mathbf{t}_{\text{miss}$ la remplace dans la s\'{e}quence en entr\'{e}e du transformer cross-modal, permettant au m\'{e}canisme d'attention d'op\'{e}rer sans modification architecturale.

\subsection{M\'{e}canisme de remplacement : masque bool\'{e}en et encodage positionnel}

La construction de la s\'{e}quence d'entr\'{e}e du transformer repose sur un masque bool\'{e}en $\mathbf{m} = (m_f, m_p)$ et un encodage positionnel diff\'{e}renci\'{e} par modalit\'{e}. On d\'{e}finit la s\'{e}quence de deux tokens :
\begin{equation}
\label{eq:sequence-tokens}
\mathbf{T} = \bigl[ \, \tilde{\mathbf{h}}_f \,; \, \tilde{\mathbf{h}}_p \, \bigr] \in \mathbb{R}^{2 \times 256},
\end{equation}
o\`{u} les tokens effectifs sont d\'{e}finis par :
\begin{equation}
\tilde{\mathbf{h}}_k = m_k \cdot \mathbf{h}_k + (1 - m_k) \cdot \mathbf{t}_{\text{miss}}, \quad k \in \{f, p\}.
\end{equation}

Un encodage positionnel apprenable par modalit\'{e} est ajout\'{e} \`{a} chaque token. On d\'{e}finit $\mathbf{e}_f, \mathbf{e}_p \in \mathbb{R}^{256}$, initialis\'{e}s par encodage sinuso\'{i}dal standard et affin\'{e}s pendant l'entra\^{i}nement. La s\'{e}quence positionn\'{e}e est :
\begin{equation}
\mathbf{T}_{\text{pos}} = \bigl[ \, \tilde{\mathbf{h}}_f + \mathbf{e}_f \,; \, \tilde{\mathbf{h}}_p + \mathbf{e}_p \, \bigr] \in \mathbb{R}^{2 \times 256}.
\end{equation}

L'entra\^{i}nement avec \emph{modality dropout} (ModDrop), introduit par Neverova \emph{et al.} \cite{neverova2016moddrop}, renforce la capacit\'{e} du syst\`{e}me \`{a} fonctionner avec des modalit\'{e}s incompl\`{e}tes. Avec une probabilit\'{e} $p_{\text{drop}} = 0{,}30$, chaque modalit\'{e} est al\'{e}atoirement masqu\'{e}e pendant l'entra\^{i}nement. Cette proc\'{e}dure force le transformer \`{a} apprendre des corr\'{e}lations cross-modales robustes tout en pr\'{e}servant la capacit\'{e} unimodale de d\'{e}gradation gracieuse.

\section{Module de Fusion par Attention Crois\'{e}e Modul\'{e}e par la Qualit\'{e}}
\label{sec:attention}

\subsection{Attention crois\'{e}e bidirectionnelle : requ\^{e}tes visage $\rightarrow$ cl\'{e}s/valeurs empreinte, et inversement}

Le module de fusion est un empilement de $L = 4$ couches de transformer \`{a} attention crois\'{e}e. Chaque couche $\ell$ maintient deux directions d'attention : la visage vers empreinte ($f \rightarrow p$) et l'empreinte vers visage ($p \rightarrow f$).

Soit $\mathbf{T}^{(\ell)} = [\mathbf{t}_f^{(\ell)} ; \mathbf{t}_p^{(\ell)}] \in \mathbb{R}^{2 \times 256}$ la s\'{e}quence en entr\'{e}e de la couche $\ell$, avec $\mathbf{t}_f^{(0)} = \tilde{\mathbf{h}}_f + \mathbf{e}_f$ et $\mathbf{t}_p^{(0)} = \tilde{\mathbf{h}}_p + \mathbf{e}_p$. Pour chaque direction d'attention, on calcule les projections requ\^{e}te, cl\'{e} et valeur :
\begin{align}
\mathbf{Q}_f^{(\ell)} &= \mathbf{t}_f^{(\ell)} \mathbf{W}_Q^{(\ell,f)}, & \mathbf{K}_p^{(\ell)} &= \mathbf{t}_p^{(\ell)} \mathbf{W}_K^{(\ell,p)}, & \mathbf{V}_p^{(\ell)} &= \mathbf{t}_p^{(\ell)} \mathbf{W}_V^{(\ell,p)}, \\
\mathbf{Q}_p^{(\ell)} &= \mathbf{t}_p^{(\ell)} \mathbf{W}_Q^{(\ell,p)}, & \mathbf{K}_f^{(\ell)} &= \mathbf{t}_f^{(\ell)} \mathbf{W}_K^{(\ell,f)}, & \mathbf{V}_f^{(\ell)} &= \mathbf{t}_f^{(\ell)} \mathbf{W}_V^{(\ell,f)},
\end{align}
o\`{u} $\mathbf{W}_Q^{(\cdot)}, \mathbf{W}_K^{(\cdot)}, \mathbf{W}_V^{(\cdot)} \in \mathbb{R}^{256 \times 256}$ sont les matrices de projection apprises.

\subsection{Modulation par qualit\'{e} : scores d'attention pond\'{e}r\'{e}s par $q_f \cdot q_p$}

La modulation par qualit\'{e} constitue le m\'{e}canisme central de l'UFM-Transformer. L'observation fondamentale, tir\'{e}e de l'analyse de l'\'{e}tat de l'art (chapitre 1), est que la modulation de qualit\'{e} appliqu\'{e}e au niveau score ne peut r\'{e}cup\'{e}rer aucune information cross-modale perdue lors de l'extraction ind\'{e}pendante des caract\'{e}ristiques. La modulation doit intervenir au niveau feature pour influencer les interactions cross-modales d\`{e}s l'apprentissage de la repr\'{e}sentation.

Pour chaque t\^{e}te $h \in \{1, \ldots, H\}$ de la direction $f \rightarrow p$, les logits d'attention sont calcul\'{e}s comme :
\begin{equation}
\label{eq:attention-logits}
\alpha_{f \rightarrow p}^{(h,\ell)} = \frac{(\mathbf{Q}_f^{(h,\ell)})(\mathbf{K}_p^{(h,\ell)})^\top}{\sqrt{d_h}} + \log(q_f \cdot q_p + \epsilon),
\end{equation}
o\`{u} $d_h = d / H = 256 / 8 = 32$ est la dimension par t\^{e}te, et $\epsilon = 10^{-6}$ \'{e}vite l'instabilit\'{e} num\'{e}rique lorsque $q_f \cdot q_p \approx 0$. Le terme additif $\log(q_f \cdot q_p)$ module les logits : si une modalit\'{e} est de faible qualit\'{e}, le produit $q_f \cdot q_p$ est r\'{e}duit, ce qui abaisse les logits d'attention et donc les poids du \emph{softmax} associ\'{e}.

Les poids d'attention sont obtenus par :
\begin{equation}
\mathbf{A}_{f \rightarrow p}^{(h,\ell)} = \text{softmax}\left(\frac{\alpha_{f \rightarrow p}^{(h,\ell)}}{\tau}\right) \in \mathbb{R}^{1 \times 1},
\end{equation}
o\`{u} $\tau = 1{,}0$ est la temp\'{e}rature du \emph{softmax}. L'attention modul\'{e}e produit le token attendu :
\begin{equation}
\hat{\mathbf{t}}_{f \leftarrow p}^{(h,\ell)} = \mathbf{A}_{f \rightarrow p}^{(h,\ell)} \cdot \mathbf{V}_p^{(h,\ell)} \in \mathbb{R}^{1 \times d_h}.
\end{equation}

La sortie multi-t\^{e}tes est la concat\'{e}nation suivie d'une projection lin\'{e}aire :
\begin{equation}
\text{MHCA}_{f \leftarrow p}^{(\ell)} = \text{Concat}\bigl[\hat{\mathbf{t}}_{f \leftarrow p}^{(1,\ell)}, \ldots, \hat{\mathbf{t}}_{f \leftarrow p}^{(H,\ell)}\bigr] \mathbf{W}_O^{(\ell,f)} \in \mathbb{R}^{1 \times 256},
\end{equation}
o\`{u} $\mathbf{W}_O^{(\ell,f)} \in \mathbb{R}^{256 \times 256}$. Le m\^{e}me calcul est effectu\'{e} pour la direction $p \rightarrow f$.

\subsection{Masquage pour modalit\'{e}s absentes : attention nulle vers/depuis modalit\'{e}s manquantes}

Lorsqu'une modalit\'{e} est manquante ($m_k = 0$), le token correspondant est $\mathbf{t}_{\text{miss}}$, et le score de qualit\'{e} est forc\'{e} \`{a} $q_k = 0$. Dans ce cas, le terme de modulation $\log(q_k + \epsilon)$ devient fortement n\'{e}gatif, ce qui annule pratiquement les poids d'attention dans la direction concern\'{e}e. Formellement, on applique un masque additif $M_{f \rightarrow p} = -\infty$ si $m_p = 0$, garantissant :
\begin{equation}
\mathbf{A}_{f \rightarrow p}^{(h,\ell)} = 0 \quad \text{si } m_p = 0.
\end{equation}
Cette propri\'{e}t\'{e} assure que le syst\`{e}me ne produit pas d'attention parasite vers une modalit\'{e} absente. Lorsque les deux modalit\'{e}s sont pr\'{e}sentes, le m\'{e}canisme produit une attention crois\'{e}e compl\`{e}te ; lorsqu'une seule est pr\'{e}sente, le transformer se comporte comme un passage avec connexion r\'{e}siduelle, pr\'{e}servant l'information disponible.

\subsection{Architecture d\'{e}taill\'{e}e : 4 couches, 8 t\^{e}tes, 256 dimensions}

Chaque couche $\ell$ du transformer cross-modal est structur\'{e}e selon le sch\'{e}ma pr\'{e}-normalisation (Pre-LN) suivant :
\begin{align}
\mathbf{t}_f'^{(\ell)} &= \mathbf{t}_f^{(\ell)} + \text{MHCA}_{f \leftarrow p}^{(\ell)}\bigl(\text{LN}(\mathbf{t}_f^{(\ell)}), \text{LN}(\mathbf{t}_p^{(\ell)})\bigr), \\
\mathbf{t}_p'^{(\ell)} &= \mathbf{t}_p^{(\ell)} + \text{MHCA}_{p \leftarrow f}^{(\ell)}\bigl(\text{LN}(\mathbf{t}_p^{(\ell)}), \text{LN}(\mathbf{t}_f^{(\ell)})\bigr), \\
\mathbf{t}_f^{(\ell+1)} &= \mathbf{t}_f'^{(\ell)} + \text{FFN}_f^{(\ell)}\bigl(\text{LN}(\mathbf{t}_f'^{(\ell)})\bigr), \\
\mathbf{t}_p^{(\ell+1)} &= \mathbf{t}_p'^{(\ell)} + \text{FFN}_p^{(\ell)}\bigl(\text{LN}(\mathbf{t}_p'^{(\ell)})\bigr),
\end{align}
o\`{u} LN d\'{e}signe la normalisation de couche \cite{ba2016layer}, et FFN est le r\'{e}seau feed-forward \`{a} deux couches avec activation GELU et facteur d'expansion 4 :
\begin{equation}
\text{FFN}(\mathbf{x}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2,
\end{equation}
avec $\mathbf{W}_1 \in \mathbb{R}^{1024 \times 256}$ et $\mathbf{W}_2 \in \mathbb{R}^{256 \times 1024}$.

Le token fusionn\'{e} final, not\'{e} $\mathbf{c} \in \mathbb{R}^{256}$, est obtenu par concatenation des tokens de sortie suivie d'une projection lin\'{e}aire :
\begin{equation}
\label{eq:token-fusionne}
\mathbf{c} = \mathbf{W}_{\text{fuse}} \cdot \text{Concat}\bigl[\mathbf{t}_f^{(L)}, \mathbf{t}_p^{(L)}\bigr] + \mathbf{b}_{\text{fuse}},
\end{equation}
o\`{u} $\mathbf{W}_{\text{fuse}} \in \mathbb{R}^{256 \times 512}$. Ce token $\mathbf{c}$ constitue la repr\'{e}sentation fusionn\'{e}e bimodale, portant l'information int\'{e}gr\'{e}e des deux modalit\'{e}s (ou de la modalit\'{e} unique disponible, augment\'{e}e du proxy token).

Le tableau \ref{tab:hyperparams-attention} r\'{e}capitule les hyperparam\'{e}tres du module d'attention.

\begin{table}[htbp]
\centering
\caption{Hyperparam\`{e}tres du module de fusion par attention crois\'{e}e}
\label{tab:hyperparams-attention}
\begin{tabular}{lcl}
\hline
\hline
Param\`{e}tre & Symbole & Valeur \\
\hline
Nombre de couches & $L$ & 4 \\
Nombre de t\^{e}tes par couche & $H$ & 8 \\
Dimension latente & $d$ & 256 \\
Dimension par t\^{e}te & $d_h = d/H$ & 32 \\
Facteur d'expansion FFN & -- & 4 \\
Dimension interm\'{e}diaire FFN & -- & 1024 \\
Temp\'{e}rature softmax & $\tau$ & 1{,}0 \\
Epsilon num\'{e}rique & $\epsilon$ & $10^{-6}$ \\
Taux de dropout (entra\^{i}nement) & -- & 0{,}1 \\
\hline
\hline
\end{tabular}
\end{table}

\subsection{Figure : d\'{e}tail du m\'{e}canisme d'attention crois\'{e}e modul\'{e}e par qualit\'{e}}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{attention_qualite_detail.pdf}
\caption{D\'{e}tail du m\'{e}canisme d'attention crois\'{e}e modul\'{e}e par qualit\'{e} pour une couche. La direction visage $\rightarrow$ empreinte est illustr\'{e}e ; la direction inverse est sym\'{e}trique. Le bloc de modulation multiplique les logits d'attention par $\log(q_f \cdot q_p)$ avant le softmax. Le masque bool\'{e}en $\mathbf{m}$ force les poids \`{a} z\'{e}ro pour les modalit\'{e}s absentes.}
\label{fig:attention-detail}
\end{figure}

\subsection{Algorithme : pseudo-code du \emph{forward pass}}

L'algorithme \ref{alg:forward} formalise le passage avant complet de l'UFM-Transformer. Les dimensions tensorielles sont indiqu\'{e}es en commentaires \`{a} chaque \'{e}tape.

\begin{algorithm}[htbp]
\caption{Forward pass de l'UFM-Transformer}
\label{alg:forward}
\begin{algorithmic}[1]
\REQUIRE Image visage $\mathbf{x}_f$ ou $\varnothing$, image empreinte $\mathbf{x}_p$ ou $\varnothing$, masque $\mathbf{m} = (m_f, m_p)$
\ENSURE Score de similarit\'{e} $s$, variance d'incertitude $\sigma^2$, scores de qualit\'{e} $(q_f, q_p)$

\STATE \textbf{\# Module d'encodage}
\IF{$m_f = 1$}
    \STATE $\mathbf{z}_f \leftarrow \mathcal{E}_f(\mathbf{x}_f)$ \hfill $\triangleright$ $\mathbb{R}^{1408 \times 7 \times 7}$
    \STATE $q_f \leftarrow \mathcal{Q}_f(\mathbf{x}_f)$ \hfill $\triangleright$ $[0, 1]$
\ELSE
    \STATE $\mathbf{z}_f \leftarrow \mathbf{0}$, $q_f \leftarrow 0$
\ENDIF

\IF{$m_p = 1$}
    \STATE $\mathbf{z}_p \leftarrow \mathcal{E}_p(\mathbf{x}_p)$ \hfill $\triangleright$ $\mathbb{R}^{512 \times 7 \times 7}$
    \STATE $q_p \leftarrow \mathcal{Q}_p(\mathbf{x}_p)$ \hfill $\triangleright$ $[0, 1]$
\ELSE
    \STATE $\mathbf{z}_p \leftarrow \mathbf{0}$, $q_p \leftarrow 0$
\ENDIF

\STATE \textbf{\# Projection sur l'hypersph\`{e}re}
\STATE $\mathbf{h}_f \leftarrow \mathcal{P}_f(\text{Flatten}(\mathbf{z}_f))$ \hfill $\triangleright$ $\mathbb{S}^{255}$
\STATE $\mathbf{h}_p \leftarrow \mathcal{P}_p(\text{Flatten}(\mathbf{z}_p))$ \hfill $\triangleright$ $\mathbb{S}^{255}$

\STATE \textbf{\# Gestion des modalit\'{e}s manquantes}
\STATE $\tilde{\mathbf{h}}_f \leftarrow m_f \cdot \mathbf{h}_f + (1 - m_f) \cdot \mathbf{t}_{\text{miss}}$
\STATE $\tilde{\mathbf{h}}_p \leftarrow m_p \cdot \mathbf{h}_p + (1 - m_p) \cdot \mathbf{t}_{\text{miss}}$

\STATE \textbf{\# Encodage positionnel}
\STATE $\mathbf{t}_f^{(0)} \leftarrow \tilde{\mathbf{h}}_f + \mathbf{e}_f$, $\mathbf{t}_p^{(0)} \leftarrow \tilde{\mathbf{h}}_p + \mathbf{e}_p$ \hfill $\triangleright$ $\mathbb{R}^{2 \times 256}$

\STATE \textbf{\# Transformer cross-modal}
\FOR{$\ell = 0$ \TO $L-1$}
    \STATE $\mathbf{Q}_f \leftarrow \text{LN}(\mathbf{t}_f^{(\ell)}) \mathbf{W}_Q^{(\ell,f)}$
    \STATE $\mathbf{K}_p \leftarrow \text{LN}(\mathbf{t}_p^{(\ell)}) \mathbf{W}_K^{(\ell,p)}$
    \STATE $\mathbf{V}_p \leftarrow \text{LN}(\mathbf{t}_p^{(\ell)}) \mathbf{W}_V^{(\ell,p)}$
    \STATE $\alpha_{f \rightarrow p} \leftarrow \frac{\mathbf{Q}_f \mathbf{K}_p^\top}{\sqrt{32}} + \log(q_f \cdot q_p + \epsilon)$
    \STATE $\mathbf{A}_{f \rightarrow p} \leftarrow \text{softmax}(\alpha_{f \rightarrow p})$ \hfill $\triangleright$ Masque nul si $m_p = 0$
    \STATE $\hat{\mathbf{t}}_f \leftarrow \mathbf{t}_f^{(\ell)} + \mathbf{A}_{f \rightarrow p} \mathbf{V}_p \mathbf{W}_O^{(\ell,f)}$
    \STATE (direction $p \rightarrow f$ sym\'{e}trique)
    \STATE $\mathbf{t}_f^{(\ell+1)} \leftarrow \hat{\mathbf{t}}_f + \text{FFN}^{(\ell)}(\text{LN}(\hat{\mathbf{t}}_f))$
    \STATE $\mathbf{t}_p^{(\ell+1)} \leftarrow \hat{\mathbf{t}}_p + \text{FFN}^{(\ell)}(\text{LN}(\hat{\mathbf{t}}_p))$
\ENDFOR

\STATE \textbf{\# Token fusionn\'{e}}
\STATE $\mathbf{c} \leftarrow \mathbf{W}_{\text{fuse}} \cdot \text{Concat}[\mathbf{t}_f^{(L)}, \mathbf{t}_p^{(L)}]$ \hfill $\triangleright$ $\mathbb{R}^{256}$

\STATE \textbf{\# T\^{e}te de similarit\'{e}}
\STATE $s \leftarrow \mathcal{S}(\mathbf{c})$ \hfill $\triangleright$ Score cosinus avec marge ArcFace

\STATE \textbf{\# T\^{e}te d'incertitude (MC-Dropout, $K=5$ passes)}
\STATE $\{\hat{s}_k\}_{k=1}^K \leftarrow \{\mathcal{U}(\mathbf{c})\}_{k=1}^K$ \hfill $\triangleright$ Dropout actif
\STATE $\bar{s} \leftarrow \frac{1}{K} \sum_k \hat{s}_k$, $\sigma^2_{\text{tot}} \leftarrow \frac{1}{K} \sum_k (\hat{s}_k - \bar{s})^2$

\RETURN $(s, \sigma^2_{\text{tot}}, q_f, q_p)$
\end{algorithmic}
\end{algorithm}

\section{T\^{e}te de Similarit\'{e} et T\^{e}te d'Incertitude}
\label{sec:similarite}

\subsection{Similarit\'{e} cosinus avec marge additive ArcFace ($m=0{,}5$, $s=30$)}

La t\^{e}te de similarit\'{e} calcule le score de correspondance entre le token fusionn\'{e} $\mathbf{c}$ et les vecteurs de poids des classes. On adopte la formulation ArcFace \cite{deng2019arcface}, qui ajoute une marge additive angulaire dans l'espace cosinus pour am\'{e}liorer la discriminabilit\'{e} inter-classe.

Soit $\mathbf{W}_j \in \mathbb{R}^{256}$ le vecteur de poids de la classe $j$ (normalis\'{e} : $\|\mathbf{W}_j\|_2 = 1$), et $\theta_j = \arccos(\mathbf{c} \cdot \mathbf{W}_j / \|\mathbf{c}\|_2)$ l'angle entre $\mathbf{c}$ et $\mathbf{W}_j$. Le logit de la classe $j$ est d\'{e}fini par :
\begin{equation}
\label{eq:arcface}
\ell_j = s \cdot \cos(\theta_j + m \cdot \mathbb{1}_{[j = y]}),
\end{equation}
o\`{u} $s = 30$ est le facteur d'\'{e}chelle, $m = 0{,}5$ la marge additive, et $y$ l'\'{e}tiquette de la v\'{e}ritable identit\'{e}. Le score de similarit\'{e} binaire entre deux \'{e}chantillons est le cosinus de leurs tokens fusionn\'{e}s respectifs :
\begin{equation}
s(\mathbf{c}, \mathbf{c}') = \frac{\mathbf{c} \cdot \mathbf{c}'}{\|\mathbf{c}\|_2 \|\mathbf{c}'\|_2}.
\end{equation}

Le choix de $s = 30$ et $m = 0{,}5$ est conforme aux valeurs standard de la litt\'{e}rature sur la reconnaissance faciale \cite{deng2019arcface} et a \'{e}t\'{e} valid\'{e} empiriquement sur notre corpus de validation. La marge $m = 0{,}5$ correspond \`{a} un \'{e}cart angulaire d'environ $28{,}6^{\circ}$, suffisant pour renforcer la s\'{e}parabilit\'{e} sans provoquer de sous-apprentissage.

\subsection{T\^{e}te d'incertitude : 5 passes avant avec dropout actif}

La quantification de l'incertitude repose sur le Monte Carlo Dropout (MC-Dropout), introduit par Gal \& Ghahramani \cite{gal2016dropout}. Cette m\'{e}thode interpr\`{e}te le dropout comme une approximation variationnelle de l'inf\'{e}rence bay\'{e}sienne, sans n\'{e}cessiter de modification architecturale. La t\^{e}te d'incertitude $\mathcal{U}$ est constitu\'{e}e de deux couches enti\`{e}rement connect\'{e}es de $256 \rightarrow 128 \rightarrow 1$ neuronnes, avec activation ReLU interm\'{e}diaire et dropout $p = 0{,}2$ entre les couches.

\`{A} l'inf\'{e}rence, $K = 5$ passes avant sont effectu\'{e}es avec le dropout actif, produisant un ensemble de pr\'{e}dictions $\{\hat{s}_k\}_{k=1}^K$. La moyenne et la variance empiriques sont calcul\'{e}es :
\begin{align}
\bar{s} &= \frac{1}{K} \sum_{k=1}^K \hat{s}_k, \\
\sigma^2_{\text{tot}} &= \frac{1}{K} \sum_{k=1}^K (\hat{s}_k - \bar{s})^2.
\end{align}

\subsection{D\'{e}composition al\'{e}atoire vs \'{e}pist\'{e}mique}

Suivant la taxonomie de Kendall \& Gal \cite{kendall2017uncertainties}, l'incertitude totale se d\'{e}compose en deux composantes distinctes. L'incertitude al\'{e}atoire (ou \emph{aleatoric}) provient du bruit inh\'{e}rent aux donn\'{e}es --- flou de l'image, occlusion partielle, bruit de capteur. L'incertitude \'{e}pist\'{e}mique (ou \emph{epistemic}) refl\`{e}te l'ignorance du mod\`{e}le, c'est-\`{a}-dire les zones de l'espace d'entr\'{e}e insuffisamment couvertes par les donn\'{e}es d'entra\^{i}nement.

Lorsque les deux modalit\'{e}s sont pr\'{e}sentes, l'incertitude al\'{e}atoire est estim\'{e}e par la variance des pr\'{e}dictions sous la m\^{e}me entr\'{e}e, l\`{a} o\`{u} le mod\`{e}le est certain de ses poids mais les donn\'{e}es sont bruit\'{e}es. L'incertitude \'{e}pist\'{e}mique est estim\'{e}e par la variance inter-passes, qui capture le d\'{e}saccord du mod\`{e}le sur ses propres param\`{e}tres. Formellement :
\begin{align}
\label{eq:decomposition-aleatoric}
\sigma^2_{\text{alea}} &= \mathbb{E}_{\mathbf{\omega}}\left[\text{Var}\bigl(s \mid \mathbf{x}, \mathbf{\omega}\bigr)\right], \\
\label{eq:decomposition-epistemic}
\sigma^2_{\text{epis}} &= \text{Var}_{\mathbf{\omega}}\left[\mathbb{E}\bigl(s \mid \mathbf{x}, \mathbf{\omega}\bigr)\right],
\end{align}
o\`{u} $\mathbf{\omega}$ d\'{e}signe les param\`{e}tres du r\'{e}seau. L'incertitude totale v\'{e}rifie la d\'{e}composition de la loi de la variance totale : $\sigma^2_{\text{tot}} = \sigma^2_{\text{alea}} + \sigma^2_{\text{epis}}$.

\subsection{Intervalle de confiance et option de rejet}

Le score de similarit\'{e} $s$ et sa variance $\sigma^2_{\text{tot}}$ permettent de construire un intervalle de confiance \`{a} $95\%$ pour la d\'{e}cision :
\begin{equation}
\label{eq:intervalle-confiance}
\text{IC}_{95\%}(s) = \left[ \bar{s} - 1{,}96 \cdot \sigma_{\text{tot}}, \; \bar{s} + 1{,}96 \cdot \sigma_{\text{tot}} \right].
\end{equation}

Le syst\`{e}me peut \'{e}mettre une option de rejet (\emph{reject option}) lorsque l'incertitude exc\`{e}de un seuil pr\'{e}d\'{e}fini $\sigma^2_{\max}$. Cette fonctionnalit\'{e}, align\'{e}e sur le principe de la pr\'{e}diction conforme \cite{balomenos2018conformal}, permet d'abstention de d\'{e}cider lorsque la confiance est insuffisante :
\begin{equation}
\label{eq:rejet}
\text{d\'{e}cision} = \begin{cases}
\text{g\'{e}nuin} & \text{si } \bar{s} > \tau \text{ et } \sigma^2_{\text{tot}} < \sigma^2_{\max}, \\
\text{imposteur} & \text{si } \bar{s} \leq \tau \text{ et } \sigma^2_{\text{tot}} < \sigma^2_{\max}, \\
\text{rejet ("ne sait pas")} & \text{si } \sigma^2_{\text{tot}} \geq \sigma^2_{\max}.
\end{cases}
\end{equation}

La valeur de $\sigma^2_{\max}$ est calibr\'{e}e sur l'ensemble de validation pour obtenir un taux de rejet cible de $5\%$ sur les paires d'identit\'{e}s connues.

\section{M\'{e}canismes d'Explicabilit\'{e}}
\label{sec:explicabilite}

\subsection{Visualisation des cartes d'attention cross-modale}

L'attention crois\'{e}e constitue une source d'explicabilit\'{e} inh\'{e}rente \`{a} l'architecture. Les poids $\mathbf{A}_{f \rightarrow p}^{(h,\ell)}$ et $\mathbf{A}_{p \rightarrow f}^{(h,\ell)}$ r\'{e}v\`{e}lent, pour chaque paire de tokens, l'importance relative accord\'{e}e \`{a} la modalit\'{e} compl\'{e}mentaire. Cette propri\'{e}t\'{e} double usage --- fusion et explication --- n'a pas \'{e}t\'{e} exploit\'{e}e dans les syst\`{e}mes biom\'{e}triques multimodaux pr\'{e}c\'{e}dents \cite{selvarani2026, yang2024authformer}.

Pour chaque couche $\ell$ et chaque t\^{e}te $h$, on visualise la matrice d'attention $\mathbf{A}^{(h,\ell)} \in \mathbb{R}^{2 \times 2}$ comme un graphe bipartite o\`{u} les n\oe uds repr\'{e}sentent les tokens visage et empreinte, et les ar\^{e}tes pond\'{e}r\'{e}es indiquent les forces d'attention. L'agr\'{e}gation sur toutes les t\^{e}tes et couches, pond\'{e}r\'{e}e par l'\'{e}cart-type de chaque t\^{e}te (crit\`{e}re de s\'{e}lectivit\'{e}), fournit une mesure globale de la contribution cross-modale :
\begin{equation}
\label{eq:attention-agregee}
\bar{\mathbf{A}} = \sum_{\ell=1}^{L} \sum_{h=1}^{H} w_h \cdot \mathbf{A}^{(h,\ell)}, \quad w_h = \frac{\sigma(\mathbf{A}^{(h,\cdot)})}{\sum_{h'} \sigma(\mathbf{A}^{(h',\cdot)})}.
\end{equation}

\subsection{Grad-CAM bimodal : r\'{e}gions faciales et minuties d'empreinte}

Pour localiser les r\'{e}gions discriminantes au niveau pixel, on combine la visualisation d'attention avec Grad-CAM \cite{selvaraju2017gradcam}. Pour le visage, on calcule la carte de saillance $\mathbf{M}_f \in \mathbb{R}^{7 \times 7}$ en r\'{e}tro-propageant le gradient du score de similarit\'{e} $s$ jusqu'\`{a} la carte de caract\'{e}ristiques $\mathbf{z}_f$ :
\begin{equation}
\mathbf{M}_f = \text{ReLU}\left( \sum_{c=1}^{1408} w_c^{(f)} \cdot \mathbf{z}_{f,c} \right), \quad w_c^{(f)} = \frac{1}{7 \times 7} \sum_{i,j} \frac{\partial s}{\partial \mathbf{z}_{f,c}^{(i,j)}}.
\end{equation}
La carte est ensuite interpol\'{e}e \`{a} la r\'{e}solution originale $224 \times 224$ par bilin\'{e}aire et superpos\'{e}e \`{a} l'image faciale.

Pour l'empreinte, le m\^{e}me calcul produit $\mathbf{M}_p \in \mathbb{R}^{7 \times 7}$. L'alignement spontan\'{e} des cartes d'attention ViT avec les r\'{e}gions riches en minuties (IoU $= 0{,}41 \pm 0{,}07$), document\'{e} par Gazi \emph{et al.} \cite{gazi2026}, garantit que les zones de forte saillance correspondent aux structures papillaires discriminantes.

La figure \ref{fig:explicabilite} illustre un exemple de visualisation bimodale compl\`{e}te, o\`{u} les r\'{e}gions faciales chaudes (yeux, nez, m\^{a}choire) sont confront\'{e}es aux zones d'empreinte actives (minuties, noyaux, deltas), avec les poids d'attention cross-modale indiquant la correspondance.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.9\textwidth]{explicabilite_bimodale.pdf}
\caption{Visualisation bimodale de l'explicabilit\'{e}. \`{A} gauche : carte Grad-CAM superpos\'{e}e \`{a} l'image faciale, avec les r\'{e}gions chaudes indiquant les zones contributives. Au centre : carte d'attention de l'empreinte align\'{e}e sur les minuties. \`{A} droite : graphe bipartite des poids d'attention cross-modale entre les deux repr\'{e}sentations.}
\label{fig:explicabilite}
\end{figure}

\subsection{Analyse des cas d'\'{e}chec par attention}

Les cartes d'attention permettent \'{e}galement d'analyser les cas d'erreur. Une fausse acceptation (\emph{false accept}) se produit lorsque le score $s$ exc\`{e}de le seuil $\tau$ pour une paire d'identit\'{e}s diff\'{e}rentes. La visualisation des poids d'attention r\'{e}v\`{e}le alors si l'erreur provient d'une sur-attention \`{a} une zone non discriminante (par exemple, le fond de l'image visage) ou d'une similarit\'{e} fortuite entre des minuties d'empreintes de doigts diff\'{e}rents. Cette capacit\'{e} de diagnostic post-hoc, h\'{e}rit\'{e}e des travaux de Lu \emph{et al.} \cite{lu2024corr rise} sur CorrRISE, est ici int\'{e}gr\'{e}e de mani\`{e}re inh\'{e}rente \`{a} l'architecture, sans n\'{e}cessiter de m\'{e}thode post-hoc externe.

\section{Fonction de Perte et Strat\'{e}gie d'Entra\^{i}nement}
\label{sec:perte}

\subsection{Perte composite : triplet + ArcFace + r\'{e}gularisation d'incertitude}

L'apprentissage de l'UFM-Transformer repose sur une fonction de perte composite combinant trois objectifs compl\'{e}mentaires :
\begin{equation}
\label{eq:perte-totale}
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ArcFace}} + \lambda_{\text{triplet}} \cdot \mathcal{L}_{\text{triplet}} + \lambda_{\sigma} \cdot \mathcal{L}_{\sigma},
\end{equation}
avec $\lambda_{\text{triplet}} = 0{,}5$ et $\lambda_{\sigma} = 0{,}1$.

La perte ArcFace, d\'{e}j\`{a} introduite en \S\ref{sec:similarite}, est la composante principale de supervision identitaire. Pour un batch de $B$ \'{e}chantillons avec leurs \'{e}tiquettes $\{y_b\}_{b=1}^B$, elle s'\'{e}crit :
\begin{equation}
\mathcal{L}_{\text{ArcFace}} = -\frac{1}{B} \sum_{b=1}^B \log \frac{e^{s \cdot \cos(\theta_{y_b} + m)}}{e^{s \cdot \cos(\theta_{y_b} + m)} + \sum_{j \neq y_b} e^{s \cdot \cos(\theta_j)}}.
\end{equation}

La perte triplet renforce la s\'{e}parabilit\'{e} des paires dans l'espace latent. Pour un anc $\mathbf{c}_a$, un positif $\mathbf{c}_p$ (m\^{e}me identit\'{e}) et un n\'{e}gatif $\mathbf{c}_n$ (identit\'{e} diff\'{e}rente), elle s'\'{e}crit :
\begin{equation}
\label{eq:perte-triplet}
\mathcal{L}_{\text{triplet}} = \frac{1}{B} \sum_{b=1}^B \max\bigl(0, \; \|\mathbf{c}_a - \mathbf{c}_p\|_2^2 - \|\mathbf{c}_a - \mathbf{c}_n\|_2^2 + \margin\bigr),
\end{equation}
o\`{u} $\margin = 0{,}3$ est la marge de s\'{e}paration. Les triplets sont \'{e}chantillonn\'{e}s selon la strat\'{e}gie \emph{semi-hard negative mining} : pour chaque ancre, on s\'{e}lectionne un n\'{e}gatif v\'{e}rifiant $\|\mathbf{c}_a - \mathbf{c}_n\|_2^2 < \|\mathbf{c}_a - \mathbf{c}_p\|_2^2 + \margin$ mais dont la distance reste sup\'{e}rieure \`{a} celle du positif.

La r\'{e}gularisation d'incertitude encadre la variance pr\'{e}dite par la t\^{e}te $\mathcal{U}$. Inspir\'{e}e par Chang \emph{et al.} \cite{chang2020data}, elle p\'{e}nalise les pr\'{e}dictions de forte incertitude sur les \'{e}chantillons bien class\'{e}s, tout en autorisant une variance \'{e}lev\'{e}e sur les \'{e}chantillons difficiles :
\begin{equation}
\label{eq:perte-sigma}
\mathcal{L}_{\sigma} = \frac{1}{B} \sum_{b=1}^B \left| \sigma^2_{\text{tot},b} - \gamma \cdot \mathbb{1}_{[s_b \cdot y_b < \tau]} \right|,
\end{equation}
o\`{u} $\gamma = 0{,}5$ est la variance cible pour les \'{e}chantillons mal class\'{e}s. Cette formulation encourage le mod\`{e}le \`{a} produire une incertitude \'{e}lev\'{e}e lorsque la d\'{e}cision est erron\'{e}e, et une incertitude faible lorsque la d\'{e}cision est correcte.

\subsection{Phase 1 : pr\'{e}-entra\^{i}nement unimodal (50 \'{e}poques, ArcFace seul)}

L'entra\^{i}nement proc\`{e}de en deux phases. La phase 1 pr\'{e}-entra\^{i}ne les encodeurs unimodaux de mani\`{e}re ind\'{e}pendante. L'objectif est de fournir \`{a} chaque branche une repr\'{e}sentation discriminante avant l'apprentissage conjoint.

\textbf{Encodeur visage.} L'EfficientNet-B2 pr\'{e}-entra\'{i}n\'{e} sur ImageNet est d\'{e}congel\'{e} progressivement. Seules les trois derniers blocs sont entra\^{i}n\'{e}s pendant les 30 premi\`{e}res \'{e}poques, puis l'ensemble du r\'{e}seau est affin\'{e} sur les 20 \'{e}poques suivantes. Le projecteur $\mathcal{P}_f$ et la t\^{e}te ArcFace sont entra\^{i}n\'{e}s conjointement avec la perte $\mathcal{L}_{\text{ArcFace}}$ seule. Le batch contient $B = 64$ images de $N = 16$ identit\'{e}s diff\'{e}rentes (4 images par identit\'{e}).

\textbf{Encodeur empreinte.} Le CNN r\'{e}siduel $\mathcal{E}_p$ est entra\^{i}n\'{e} from scratch avec le m\^{e}me protocole. Le batch contient $B = 64$ images de $N = 16$ identit\'{e}s. Les augmentations appliqu\'{e}es sont : rotation al\'{e}atoire $\mathcal{U}[-15^{\circ}, +15^{\circ}]$, ajustement de contraste $\mathcal{U}[0{,}8, 1{,}2]$, et ajout de bruit gaussien $\mathcal{N}(0, 0{,}01^2)$.

\textbf{Estimateurs de qualit\'{e}.} Les estimateurs $\mathcal{Q}_f$ et $\mathcal{Q}_p$ sont entra\^{i}n\'{e}s conjointement avec leurs encodeurs respectifs, supervis\'{e}s indirectement par la perte ArcFace. L'hypoth\`{e}se sous-jacente est que la qualit\'{e} apprise doit corrr\'{e}ler avec la difficult\'{e} de classification, un principe valid\'{e} par les travaux sur DUL \cite{chang2020data} et CR-FIQA \cite{boutros2023crfiqa}.

\subsection{Phase 2 : fine-tuning joint (100 \'{e}poques, dropout modalit\'{e}s 30\%, perte compl\`{e}te)}

La phase 2 assemble l'architecture compl\`{e}te et entra\^{i}ne tous les modules conjointement avec la perte composite (\ref{eq:perte-totale}). Les hyperparam\`{e}tres de cette phase sont r\'{e}sum\'{e}s dans le tableau \ref{tab:hyperparams-training}.

\begin{table}[htbp]
\centering
\caption{Hyperparam\`{e}tres de la phase 2 d'entra\^{i}nement (fine-tuning joint)}
\label{tab:hyperparams-training}
\begin{tabular}{lcl}
\hline
\hline
Param\`{e}tre & Symbole & Valeur \\
\hline
Nombre d'\'{e}poques & -- & 100 \\
Taille du batch & $B$ & 128 \\
Identit\'{e}s par batch & $N$ & 32 \\
\'{E}chantillons par identit\'{e} & -- & 4 \\
Optimiseur & -- & AdamW \cite{loshchilov2018adamw} \\
Taux d'apprentissage initial & $\eta_0$ & $10^{-4}$ \\
Taux d'apprentissage (backbones) & $\eta_{\text{back}}$ & $10^{-5}$ \\
Poids d\'{e}croissance & $\beta_1, \beta_2$ & 0{,}9, 0{,}999 \\
D\'{e}croissance de r\'{e}gularisation & $\lambda_{\text{WD}}$ & $10^{-2}$ \\
Planification & -- & Cosine annealing avec warm-up \\
\'{E}poques de warm-up & -- & 10 \\
Gradient clipping & -- & $\|\mathbf{g}\|_2 \leq 1{,}0$ \\
Dropout modalit\'{e}s (ModDrop) & $p_{\text{drop}}$ & 0{,}30 \\
Dropout interne & -- & 0{,}10 \\
\hline
\hline
\end{tabular}
\end{table}

La planification du taux d'apprentissage suit un profil de cosinus avec p\'{e}riode de chauffe (\emph{warm-up}) :
\begin{equation}
\label{eq:cosine-schedule}
\eta(t) = \begin{cases}
\eta_0 \cdot \frac{t}{T_{\text{warm}}} & \text{si } t < T_{\text{warm}}, \\
\eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warm}}}{T_{\text{max}} - T_{\text{warm}}} \pi\right)\right) & \text{sinon},
\end{cases}
\end{equation}
o\`{u} $T_{\text{warm}} = 10$ \'{e}poques, $T_{\max} = 100$ \'{e}poques, et $\eta_{\min} = 10^{-6}$. Le gradient clipping \`{a} $\|\mathbf{g}\|_2 \leq 1{,}0$ stabilise l'entra\^{i}nement du transformer, particuli\`{e}rement lorsque le dropout modalit\'{e} g\'{e}n\`{e}re des configurations d'entr\'{e}e inhabituelles.

\subsection{Optimisation : AdamW, cosine annealing, gradient clipping}

L'optimiseur AdamW \cite{loshchilov2018adamw} est pr\'{e}f\'{e}r\'{e} \`{a} Adam standard car il d\'{e}couple la d\'{e}croissance de poids (\emph{weight decay}) de la mise \`{a} jour du gradient de momentum. Cette propri\'{e}t\'{e} s'av\`{e}re essentielle pour le transformer cross-modal, o\`{u} les matrices d'attention requi\`{e}rent une r\'{e}gularisation diff\'{e}renci\'{e}e des couches feed-forward. La d\'{e}croissance de poids $\lambda_{\text{WD}} = 10^{-2}$ est appliqu\'{e}e \`{a} l'ensemble des param\`{e}tres except\'{e}s les biais et les termes de normalisation de couche.

\subsection{Algorithme : boucle d'entra\^{i}nement compl\`{e}te avec deux phases}

L'algorithme \ref{alg:training} pr\'{e}sente la boucle d'entra\^{i}nement compl\`{e}te. Les deux phases y sont explicit\'{e}es avec leurs conditions respectives.

\begin{algorithm}[htbp]
\caption{Boucle d'entra\^{i}nement compl\`{e}te de l'UFM-Transformer}
\label{alg:training}
\begin{algorithmic}[1]
\REQUIRE Donn\'{e}es d'entra\^{i}nement $\mathcal{D} = \{(\mathbf{x}_f^{(i)}, \mathbf{x}_p^{(i)}, y^{(i)})\}$, hyperparam\`{e}tres
\ENSURE Param\`{e}tres entra\^{i}n\'{e}s $\theta^*$

\STATE \textbf{--- Phase 1 : Pr\'{e}-entra\^{i}nement unimodal ---}
\STATE Initialiser $\mathcal{E}_f$ depuis ImageNet, $\mathcal{E}_p$ al\'{e}atoirement
\STATE Initialiser $\mathcal{Q}_f, \mathcal{Q}_p, \mathcal{P}_f, \mathcal{P}_p$, t\^{e}tes ArcFace

\FOR{$\text{\'{e}poque} = 1$ \TO $50$}
    \STATE \textbf{\# Branche visage}
    \FOR{each batch $(\mathbf{x}_f, y) \sim \mathcal{D}_f$}
        \STATE $\mathbf{z}_f \leftarrow \mathcal{E}_f(\mathbf{x}_f)$
        \STATE $\mathbf{h}_f \leftarrow \mathcal{P}_f(\text{Flatten}(\mathbf{z}_f))$
        \STATE $\mathcal{L} \leftarrow \mathcal{L}_{\text{ArcFace}}(\mathbf{h}_f, y)$
        \STATE R\'{e}tro-propagation et mise \`{a} jour AdamW($\eta = 10^{-4}$)
    \ENDFOR
    \STATE \textbf{\# Branche empreinte}
    \FOR{each batch $(\mathbf{x}_p, y) \sim \mathcal{D}_p$}
        \STATE $\mathbf{z}_p \leftarrow \mathcal{E}_p(\mathbf{x}_p)$
        \STATE $\mathbf{h}_p \leftarrow \mathcal{P}_p(\text{Flatten}(\mathbf{z}_p))$
        \STATE $\mathcal{L} \leftarrow \mathcal{L}_{\text{ArcFace}}(\mathbf{h}_p, y)$
        \STATE R\'{e}tro-propagation et mise \`{a} jour AdamW($\eta = 10^{-4}$)
    \ENDFOR
\ENDFOR

\STATE \textbf{--- Phase 2 : Fine-tuning joint ---}
\STATE Assembler l'architecture compl\`{e}te : $\mathcal{E}_f, \mathcal{E}_p, \mathcal{Q}_f, \mathcal{Q}_p, \mathcal{P}_f, \mathcal{P}_p$, transformer, t\^{e}tes
\STATE Initialiser $\mathbf{t}_{\text{miss}} \sim \mathcal{N}(\mathbf{0}, 0{,}02^2 \cdot \mathbf{I})$

\FOR{$\text{\'{e}poque} = 1$ \TO $100$}
    \STATE $\eta \leftarrow \text{CosineSchedule}(\text{\'{e}poque}, \eta_0 = 10^{-4}, T_{\text{warm}} = 10)$
    \FOR{each batch $(\mathbf{x}_f, \mathbf{x}_p, y) \sim \mathcal{D}$}
        \STATE \textbf{\# Modality dropout}
        \FOR{$b = 1$ \TO $B$}
            \STATE Tirage $u_f, u_p \sim \mathcal{U}[0,1]$
            \STATE $m_f^{(b)} \leftarrow \mathbb{1}_{[u_f > p_{\text{drop}}]}$, $m_p^{(b)} \leftarrow \mathbb{1}_{[u_p > p_{\text{drop}}]}$
            \STATE Si $m_f^{(b)} = m_p^{(b)} = 0$ : forcer $m_f^{(b)} \leftarrow 1$ (au moins une modalit\'{e})
        \ENDFOR
        \STATE $(s, \sigma^2, q_f, q_p) \leftarrow \text{UFM-Forward}(\mathbf{x}_f, \mathbf{x}_p, \mathbf{m})$
        \STATE \textbf{\# Perte composite}
        \STATE $\mathcal{L} \leftarrow \mathcal{L}_{\text{ArcFace}} + 0{,}5 \cdot \mathcal{L}_{\text{triplet}} + 0{,}1 \cdot \mathcal{L}_{\sigma}$
        \STATE R\'{e}tro-propagation
        \STATE $\mathbf{g} \leftarrow \text{clip}_{\text{norm}}(\mathbf{g}, 1{,}0)$
        \STATE Mise \`{a} jour AdamW($\eta$, $\lambda_{\text{WD}} = 10^{-2}$)
    \ENDFOR
    \STATE \textbf{\# Validation}
    \STATE \'{E}valuer TAR@FAR$=10^{-4}$, EER, AUC sur ensemble de validation
    \STATE Sauvegarder meilleur mod\`{e}le selon TAR@FAR$=10^{-4}$
\ENDFOR

\RETURN $\theta^* \leftarrow$ meilleurs param\`{e}tres sur validation
\end{algorithmic}
\end{algorithm}

Le nombre total de param\`{e}tres entra\^{i}nables de l'UFM-Transformer est de \textcolor{red}{\textbf{XX,X M}}. % INS\'{E}RER VALEUR R\'{E}ELLE APR\`{E}S EX\'{E}CUTION
Les encodeurs $\mathcal{E}_f$ et $\mathcal{E}_p$ concentrent la majorit\'{e} de la capacit\'{e} (respectivement 9,2 M et 11,2 M de param\`{e}tres), tandis que le module de fusion transformer ajoute seulement \textcolor{red}{\textbf{X,X M}} de param\`{e}tres suppl\'{e}mentaires, ce qui en fait une solution compacte au regard des architectures transformer biom\'{e}triques r\'{e}centes \cite{grosz2024afrnet, talreja2021deephashing}.

Ce chapitre a expos\'{e} l'int\'{e}gralit\'{e} de l'architecture UFM-Transformer, depuis la formulation math\'{e}matique du probl\`{e}me jusqu'\`{a} la boucle d'entra\^{i}nement compl\`{e}te. Chaque module a \'{e}t\'{e} justifi\'{e} par rapport aux travaux ant\'{e}rieurs, et les hyperparam\`{e}tres critiques ont \'{e}t\'{e} explicit\'{e}s avec leurs valeurs. Le chapitre suivant proc\`{e}de \`{a} l'\'{e}valuation empirique de cette architecture sur les corpus biom\'{e}triques de r\'{e}f\'{e}rence.
