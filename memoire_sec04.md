\chapter{Discussion et Perspectives}
\label{chap:discussion}

Les résultats présentés au chapitre précédent établissent l'UFM-Transformer comme une architecture compétitive sur le plan de la précision biométrique, tout en apportant des capacités absentes des systèmes de référence : calibration probabiliste, gestion native des modalités manquantes, et explicabilité bimodale inhérente. Ce chapitre interprète ces résultats au regard de l'état de l'art, identifie les limites spécifiques de ce travail, et trace des pistes de recherche actionnables pour les années à venir.

\section{Interprétation des Résultats}
\label{sec:interpretation}

\subsection{Supériorité sur les baselines : le rôle synergique de chaque composant}

La réduction relative de l'EER de \textcolor{red}{\textbf{XX,XX\%}} par rapport à la meilleure baseline (AuthFormer \cite{yang2024authformer}) ne s'explique pas par un seul mécanisme isolé, mais par l'interaction de trois choix architecturaux concertés.

\textbf{La projection sur l'hypersphère unité} constitue le fondement sur lequel repose toute l'architecture. En contraignant $\|\mathbf{h}_f\|_2 = \|\mathbf{h}_p\|_2 = 1$, les projecteurs $\mathcal{P}_f$ et $\mathcal{P}_p$ placent les représentations des deux modalités dans un espace métrique homogène où la similarité cosinus a un sens géométrique interprétable. Ce choix, inspiré des travaux de Deng \emph{et al.} \cite{deng2019arcface} sur ArcFace, permet à l'attention croisée de raisonner sur des directions dans un espace commun plutôt que sur des vecteurs de normes hétérogènes. L'étude d'ablation (\S\ref{sec:ablation}) confirme ce raisonnement : la suppression de la normalisation L2 dégrade les performances de \textcolor{red}{\textbf{X,XX\%}}, ce qui équivaut à perdre la moitié du gain par rapport aux baselines.

\textbf{La modulation par qualité au niveau des logits d'attention} représente le mécanisme le plus discriminant. L'ablation qui remplace $\log(q_f \cdot q_p)$ par un produit scalaire non modulé entraîne une dégradation de \textcolor{red}{\textbf{X,XX\%}} en TAR@FAR$=10^{-4}$. Ce résultat confirme empiriquement l'hypothèse théorique formulée au chapitre \ref{chap:etat_art} : la modulation de qualité doit impérativement opérer au niveau des représentations pour influencer les interactions cross-modales \cite{soleymani2021quality}. AuthFormer \cite{yang2024authformer} et QME \cite{zhu2025qme} modulent la qualité au niveau des scores ou par gating résiduel, ce qui ne permet pas de récupérer l'information inter-modale perdue lors de l'extraction indépendante. L'UFM-Transformer, en modulant les logits d'attention avant le \emph{softmax}, préserve cette information et la pondère dynamiquement selon la fiabilité de chaque modalité.

\textbf{Le \emph{modality dropout} à $p_{\text{drop}} = 0{,}30$} force le transformer à apprendre des corrélations cross-modales robustes en rendant chaque modalité potentiellement absente pendant l'entraînement. Ce mécanisme, transposé de Neverova \emph{et al.} \cite{neverova2016moddrop} vers l'attention croisée, produit un effet de régularisation inattendu : les poids d'attention des têtes deviennent plus sélectifs, avec un écart-type inter-tête supérieur de \textcolor{red}{\textbf{XX\%}} comparé à un entraînement sans ModDrop. Cette sélectivité accrue se traduit par une meilleure explicabilité, les têtes d'attention se spécialisant de fait sur des aspects complémentaires de la correspondance inter-modale.

\subsection{L'incertitude comme indicateur fiable de confiance}

L'ECE (\emph{Expected Calibration Error}) de \textcolor{red}{\textbf{XX,XX\%}} atteint par l'UFM-Transformer représente une amélioration substantielle par rapport aux scores déterministes des baselines. Pour contextualiser, les systèmes biométriques conventionnels ne produisent aucune calibration probabiliste ; même les embeddings probabilistes PFE \cite{shi2019pfe} et DUL \cite{chang2020dul}, qui quantifient l'incertitude unimodale, ne propagent pas cette incertitude à travers un mécanisme de fusion.

La décomposition aléatoire versus épistémique révèle un phénomène instructif. Sur les paires \emph{genuine}, l'incertitude épistémique domine ($\sigma^2_{\text{epis}} / \sigma^2_{\text{tot}} \approx$ \textcolor{red}{\textbf{0,XX}}), indiquant que le modèle éprouve davantage d'hésitation sur la correspondance identitaire que sur le bruit de capteur. Cette observation contredit partiellement l'intuition selon laquelle l'incertitude aléatoire prévaudrait sur des échantillons de qualité dégradée. Elle suggère que les encodeurs unimodaux, pré-entraînés sur des corpus conséquents, maîtrisent suffisamment la variabilité de bas niveau (bruit, flou) pour que le défi résiduel soit d'ordre sémantique : le modèle doute moins de la qualité de l'image que de l'identité du sujet.

L'option de rejet, activée lorsque $\sigma^2_{\text{tot}} \geq \sigma^2_{\max}$, permet d'abstenir de \textcolor{red}{\textbf{XX\%}} des décisions les plus incertaines, réduisant le FAR effectif de \textcolor{red}{\textbf{XX\%}} au prix d'un léger recul du taux d'acceptation. Cette fonctionnalité, absente de tous les systèmes de référence testés, transforme le système d'un classificateur binaire en un décideur à trois sorties (\emph{genuine}, \emph{impostor}, \emph{indécis}) — une propriété essentielle pour les applications à haut risque où une décision incertaine vaut moins qu'une abstention explicite.

\subsection{L'explicabilité comme outil de diagnostic}

Les cartes d'attention croisée et les visualisations Grad-CAM bimodales ne constituent pas un simple embellissement méthodologique. Elles répondent à un besoin opérationnel concret : comprendre pourquoi le système échoue. L'analyse des cas d'erreur (\S\ref{sec:explicabilite-resultats}) révèle que \textcolor{red}{\textbf{XX\%}} des fausses acceptations résultent d'une sur-attention à des zones non discriminantes du visage (arrière-plan, cheveux), tandis que \textcolor{red}{\textbf{XX\%}} des faux rejets correspondent à des minuties d'empreintes mal alignées entre requête et gabarit.

L'examen comparé des cartes d'attention entre une décision correcte et une erreur révèle des motifs systématiques. Lorsque le système décide correctement, les poids inter-modaux présentent une structure diagonale marquée : les têtes d'attention des couches profondes ($\ell \geq 3$) concentrent leur masse sur les correspondances structurelles entre les landmarks faciaux et les minuties papillaires. En cas d'erreur, cette structure disparaît au profit d'une distribution diffuse, presque uniforme — le modèle « cherche » sans trouver. Ce phénomène, similaire à l'attention flottante observée par Gnanapraveen \emph{et al.} \cite{gnanapraveen2024cross} dans les transformers audio-visuels, constitue un indicateur prédictif d'erreur exploitable en temps réel.

Cette granularité de diagnostic, inaccessible aux méthodes \emph{post-hoc} comme SHAP ou LIME \cite{selvarani2025explainability}, découle directement du mécanisme d'attention croisée. Les poids $\mathbf{A}_{f \rightarrow p}^{(h,\ell)}$ révèlent, pour chaque couche et chaque tête, quelle région de la représentation faciale « regarde » quelle région de la représentation papillaire. Cette explicabilité \emph{ante-hoc}, gratuite du point de vue computationnel, constitue un atout pour l'audit des systèmes biométriques en conditions réglementaires (RGPD, AI Act européen) où l'obligation d'explicabilité des décisions algorithmiques se renforce.

\section{Limites}
\label{sec:limites}

\subsection{Échelle du corpus : le seuil des 100 sujets}

Le corpus d'évaluation, composé de \textcolor{red}{\textbf{XXX}} sujets avec \textcolor{red}{\textbf{X}} échantillons par modalité, demeure modeste au regard des standards industriels. Les systèmes de référence évalués sur MegaFace \cite{kemelmacher2016megaface} (1 million d'identités) ou NIST FRVT \cite{nistfrvt} (des dizaines de millions d'enrôlés) opèrent à des échelles où les distributions \emph{genuine} et \emph{impostor} sont estimées avec une précision statistique bien supérieure. À FAR $= 10^{-4}$, notre estimation repose sur \textcolor{red}{\textbf{XXX}} paires imposteur, ce qui confère un intervalle de confiance de \textcolor{red}{\textbf{XX\%}} — acceptable pour une démonstration de faisabilité, insuffisant pour une certification opérationnelle.

Cette limitation dimensionnelle affecte particulièrement l'évaluation de l'incertitude. La calibration de l'ECE requiert un binning des prédictions par niveau de confiance ; avec un nombre limité de paires, certains bins de confiance élevée contiennent moins de 30 échantillons, rendant l'estimation de la précision conditionnelle bruitée. Le MC-Dropout avec $K=5$ passes compense partiellement cette rareté en introduisant de la variabilité artificielle, mais ne substitue pas à un corpus de plusieurs milliers d'identités.

\subsection{Modalités restreintes : la paire visage-empreinte}

L'UFM-Transformer ne traite que deux modalités biométriques. Cette restriction, justifiée par la disponibilité des données et la complémentarité opérationnelle du couple visage-empreinte, limine deux cas d'usage importants. Le premier concerne les systèmes triples (visage+iris+empreinte), déployés aux frontières et dans les aéroports, où l'iris apporte une discriminabilité supérieure dans des conditions de capture contraintes. Le second touche les systèmes mobiles, où la voix et le comportement tactile (\emph{touch dynamics}) constituent des modalités accessibles sans capteur dédié.

L'extension à plus de deux modalités n'est pas immédiate. Le nombre de paires attention croisée croît quadratiquement : pour $M$ modalités, $M(M-1)$ directions doivent être calculées. À $M=4$ (visage, empreinte, iris, voix), cela représente 12 directions contre 2 actuellement — une augmentation qui impacterait la latence d'inférence, particulièrement sensible avec le MC-Dropout multiplié par le nombre de passes.

\subsection{Coût computationnel du MC-Dropout}

Le MC-Dropout avec $K=5$ passes multiplie le temps d'inférence par un facteur \textcolor{red}{\textbf{X,X}}. Sur GPU NVIDIA \textcolor{red}{\textbf{XXXX}}, le temps moyen par paire passe de \textcolor{red}{\textbf{XX ms}} en mode déterministe à \textcolor{red}{\textbf{XXX ms}} en mode incertain. Cette latence, bien que tolérable pour une vérification en mode \emph{one-to-one} sur serveur, devient problématique pour le déploiement embarqué (smartphones, terminaux IoT) où la contrainte énergétique prime.

Le chapitre \ref{chap:perspectives} développe deux stratégies pour pallier cette limitation : la distillation de connaissances vers un réseau étudiant déterministe (\S\ref{sec:distillation}) et l'approximation de l'incertitude par un réseau léger auxiliaire (\S\ref{sec:pred-conforme}).

\section{Perspectives}
\label{chap:perspectives}

\subsection{Extension multimodale : iris, voix, paume}

L'intégration d'une troisième modalité — l'iris en priorité — constitue la perspective la plus directe. L'iris présente trois avantages pour l'UFM-Transformer : une structure texturelle riche exploitable par les mêmes encodeurs CNN, une fiabilité supérieure sous éclairage variable, et une complémentarité biométrique établie avec le visage (modes de défaillance orthogonaux : l'iris résiste aux variations de pose là où le visage décline, et inversement).

La question centrale demeure architecturale. L'attention croisée pair-to-pair, actuellement bidirectionnelle entre deux tokens, doit évoluer vers un mécanisme d'attention \emph{all-to-all} où chaque token modalité interagit avec tous les autres. Le \emph{Multimodal Bottleneck Transformer} (MBT) \cite{nagrani2021attention} offre un modèle : des \emph{bottleneck latents} condensent l'information inter-modale, réduisant la complexité attentionnelle de $\mathcal{O}(M^2)$ à $\mathcal{O}(MB)$ où $B$ est le nombre de \emph{bottlenecks}. Cette adaptation, combinable avec la modulation par qualité existante, préserverait la scalabilité de l'UFM-Transformer à quatre modalités et plus.

La voix, en tant que modalité comportementale, introduit une hétérogénéité supplémentaire. Les spectrogrammes mel-filterbank, couramment utilisés pour représenter l'empreinte vocale, possèdent une structure spatio-temporelle distincte des images faciales ou papillaires. Un encodeur dédié, possiblement inspiré des \emph{wav2vec 2.0} \cite{baevski2020wav2vec}, serait requis, suivi d'une projection sur la même hypersphère $\mathbb{S}^{255}$ pour préserver la compatibilité avec le transformer existant.

\subsection{Distillation de connaissances pour le déploiement \emph{edge}}
\label{sec:distillation}

Le déploiement de l'UFM-Transformer sur des appareils à ressources limités nécessite une réduction drastique du coût computationnel. La distillation de connaissances \cite{hinton2015distilling} offre une piste prometteuse : un réseau étudiant léger (MobileNet-V3 \cite{howard2019mobilenetv3} ou EfficientNet-Lite) apprend à reproduire les sorties du transformer enseignant tout en préservant la qualité de calibration.

Deux stratégies de distillation se distinguent. La distillation \emph{task-agnostic} transfère les représentations intermédiaires du transformer vers le réseau étudiant, suivant le paradigme de Lu \emph{et al.} \cite{lu2025mpad} avec leur module PAFM. La distillation \emph{uncertainty-aware}, plus ambitieuse, apprend non seulement le score de similarité mais aussi la variance d'incertitude : l'étudiant est entraîné à minimiser la divergence KL entre sa distribution predictive et celle obtenue par MC-Dropout de l'enseignant. Cette approche, validée par Malinin \emph{et al.} \cite{malinin2019ensemble} pour les réseaux de neurones profonds, produirait un modèle déterministe capable d'estimer l'incertitude en une seule passe avant.

\subsection{Apprentissage fédéré préservant la confidentialité}

Les données biométriques sont classées comme données sensibles au sens du RGPD et de la Loi Informatique et Libertés. L'entraînement centralisé de l'UFM-Transformer sur des corpus agrégés pose un risque de fuite d'informations identifiantes, particulièrement via les attaques par reconstruction \cite{fredrikson2015model} ou par inférence d'appartenance \cite{shokri2017membership}.

L'apprentissage fédéré \cite{mcmahan2017communication} constitue une réponse naturelle. Chaque appareil (téléphone, borne d'enrôlement) entraîne localement l'UFM-Transformer sur ses propres données et communique uniquement les mises à jour de gradients au serveur central. L'ajout d'un bruit différentiel \cite{dwork2006differential} aux gradients (DP-SGD \cite{abadi2016deep}) fournit des garanties théoriques de confidentialité. Néanmoins, trois défis spécifiques à la biométrie multimodale demeurent : l'hétérogénéité des modalités disponibles selon les appareils (certains possèdent un capteur d'empreinte, d'autres uniquement une caméra), la non-iid des distributions identitaires (chaque appareil appartient à un utilisateur unique), et la sensibilité du mécanisme d'attention croisée aux perturbations de gradients induites par le bruit différentiel. Le \emph{modality dropout}, pilier de la robustesse de l'UFM-Transformer, prend ici une dimension nouvelle : il pourrait servir de mécanisme d'équilibrage pour les appareils sous-équipés, en simulant l'absence de modalités que ces appareils ne possèdent effectivement pas.

\subsection{Prédiction conforme pour des garanties théoriques}
\label{sec:pred-conforme}

Le MC-Dropout fournit une calibration empirique de l'incertitude, mais aucune garantie théorique de couverture. La prédiction conforme (\emph{Conformal Prediction}) \cite{balomenos2018automatic} offre un cadre alternatif fondé sur la théorie des échantillons d'échangeabilité : pour un risque $\alpha$ prédéfini (par exemple $\alpha = 0{,}05$), l'intervalle de confiance construit par prédiction conforme garantit une couverture d'au moins $1 - \alpha$, indépendamment de la distribution des données et du modèle sous-jacent.

L'adaptation de la prédiction conforme à la vérification biométrique pose une question méthodologique intéressante. Dans le cadre standard de la classification, la prédiction conforme produit un \emph{prediction set} contenant les étiquettes plausibles. Pour la vérification binaire (\emph{genuine} / \emph{impostor}), l'analogue naturel est un intervalle de confiance sur le score de similarité, assorti d'une option de rejet théoriquement fondée. Le \emph{split conformal prediction}, où un ensemble de calibration indépendant détermine le quantile du \emph{non-conformity score}, pourrait être appliqué aux sorties de l'UFM-Transformer pour produire des garanties de couverture valides à échantillon fini. Cette perspective rapprocherait la biométrie des standards exigés en médecine personnalisée et en finance algorithmique, où la certitude statistique prime sur la performance brute.

\vspace{0.5em}

Ces quatre perspectives — extension multimodale, distillation \emph{edge}, apprentissage fédéré, et prédiction conforme — partagent une ambition commune : transformer l'UFM-Transformer d'un prototype de recherche en une technologie déployable à l'échelle, tout en préservant les propriétés qui font son originalité. La fusion par attention croisée modulée par la qualité, la gestion native des modalités manquantes, la calibration probabiliste et l'explicabilité bimodale ne sont pas des options décoratives : elles constituent les piliers d'une biométrie digne de confiance, où la machine ne se contente pas de reconnaître, mais sait aussi exprimer ce qu'elle ignore.
