\chapter{État de l'Art}
\label{chap:etat_art}

La biométrie multimodale, et spécifiquement la fusion du visage et de l'empreinte digitale, constitue l'un des dispositifs d'identification les plus déployés à l'échelle planétaire — des postes frontières aux terminaux mobiles. Pourtant, les systèmes contemporains peinent à concilier quatre exigences pourtant indissociables d'une authentification digne de confiance : une fusion représentationnelle riche, la robustesse aux modalités dégradées ou absentes, la calibration probabiliste des scores de similarité, et l'explicabilité des décisions. Ce chapitre procède à une revue systématique de la littérature organisée en cinq thèmes, et identifie de manière circonstanciée le gap scientifique que l'architecture UFM-Transformer comble.

\section{Architectures de Fusion Profonde pour la Biométrie Multimodale}
\label{sec:fusion}

La fusion multimodale en biométrie s'articule traditionnellement autour de trois stratégies — précoce, tardive et hybride — dont les propriétés informationnelles et les vulnérabilités diffèrent radicalement. La compréhension de ces différences est essentielle pour situer l'originalité de l'approche proposée.

\subsection{Fusion précoce : concaténation et espaces d'embedding joints}

La fusion précoce, ou \textit{feature-level fusion}, opère sur les représentations intermédiaires des réseaux profonds en concaténant, additionnant ou multipliant les descripteurs extrait de chaque modalité. Elle domine la littérature récente : sur 45 études examinées entre 2020 et 2026, 25 adoptent cette stratégie \cite{soleymani2021quality}. L'attrait théorique est évident — combiner les modalités dans l'espace des caractéristiques préserve la richesse informationnelle et permet au réseau d'apprendre des interactions croisées non linéaires.

Soleymani \textit{et al.} \cite{soleymani2018generalized} ont posé les bases de la fusion bilinéaire profonde pour la biométrie multimodale en proposant le \textit{Generalized Compact Bilinear} (GCB) fusion, qui capture les interactions multiplicatives entre descripteurs de modalités via une projection \textit{tensor sketch}. Le GCB surpasse systématiquement la concaténation linéaire sur les bases CMU Multi-PIE, BioCop et BIOMDATA, tout en réduisant drastiquement le nombre de paramètres. Les mêmes auteurs étendent ensuite cette approche par une abstraction multi-niveaux, fusionnant les caractéristiques issues de plusieurs couches convolutionnelles de CNNs modali-spécifiques et atteignant 99,34\% de précision sur la base BioCop pour la triple modalité visage+iris+empreinte \cite{soleymani2019multi}.

Les réseaux \textit{cross-stitch} de Misra \textit{et al.} \cite{misra2016cross}, initialement conçus pour l'apprentissage multi-tâche, fournissent un mécanisme adaptatif de partage de représentations en apprenant des combinaisons linéaires des sorties de couches entre réseaux modali-spécifiques. Ruder \textit{et al.} \cite{ruder2019latent} généralisent cette idée via les \textit{sluice networks}, qui apprennent non seulement quelles couches et sous-espaces partager, mais également où les représentations les plus discriminantes se forment — réduisant l'erreur de 12,8\% par rapport aux modèles uni-tâche.

La limitation fondamentale de la fusion précoce réside dans sa fragilité face à la dégradation d'une modalité. Lorsqu'une empreinte est bruitée ou qu'un visage est partiellement occlus, les caractéristiques corrompues se propagent directement dans la représentation jointe, compromettant l'intégralité de l'embedding fusionné. Cette vulnérabilité structurelle, soulignée par de multiples travaux \cite{liang2024deep, soleymani2021quality}, motive l'intégration de mécanismes de pondération dynamique ou de gating attentionnel — thèmes développés en \ref{sec:attention}.

\subsection{Fusion tardive : combinaison au niveau des scores}

La fusion tardive, ou \textit{score-level fusion}, constitue la stratégie la plus répandue en déploiement opérationnel. Chaque modalité traverse un pipeline indépendant (extraction, comparaison, production d'un score de similarité), et les scores résultants sont combinés par des règles fixes (somme, produit, min, max) ou par des modèles appris \cite{nandakumar2008likelihood}.

Nandakumar \textit{et al.} \cite{nandakumar2008likelihood} ont établi le cadre théorique dominant en proposant une fusion par ratio de vraisemblance reposant sur l'estimation de densités de mélange gaussiennes pour les distributions de scores \textit{genuine} et \textit{impostor}. Leur approche gère explicitement les scores à échelles arbitraires, corrélés, et intègre les mesures de qualité. Préalablement, Jain \textit{et al.} \cite{jain2005score} avaient systématisé les techniques de normalisation (min-max, z-score, tanh) et démontré que la règle de somme appliquée à des scores z-normalisés atteignait 98,6\% de GAR à 0,1\% de FAR sur des modalités visage, empreinte et géométrie de la main.

Cette élégance modulaire dissimule une limitation informationnelle irréductible. Dass \textit{et al.} \cite{dass2005principled} avaient déjà formalisé le cadre optimal de fusion par ratio de vraisemblance, démontrant que la règle du produit est optimale sous hypothèse d'indépendance tandis que la règle de la somme est plus robuste aux erreurs d'estimation. Kittler \textit{et al.} avaient quant à établi le cadre théorique des règles de combinaison fixes. Pourtant, ces règles présupposent une fiabilité approximativement égale des modalités, hypothèse systématiquement violée en conditions réelles où une empreinte peut être humide ou un visage partiellement éclairé.

Liang \textit{et al.} \cite{liang2024deep} caractérisent la fusion décisionnelle comme souffrant de «\,limitations évidentes\,» : ses performances sont «\,contraintes par une seule modalité\,» et elle est «\,moins efficace pour capturer les relations profondes entre modalités\,». Au niveau du score, la topologie de l'espace de caractéristiques croisées a été irréversiblement perdue — la fusion ne peut plus exploiter les corrélations inter-modales qui existaient dans l'espace des représentations. Les règles de combinaison fixes (somme, produit) présupposent une fiabilité approximativement égale des modalités, hypothèse systématiquement violée en conditions réelles.

Les travaux récents tentent de pallier ces limitations par des approches adaptatives. QMagFace \cite{terhorst2023qmagface} introduit une fonction de comparaison sensible à la qualité exploitant la linéarité entre qualité et score de similarité. Néanmoins, même ces méthodes sophistiquées opèrent sur des scores pré-calculés et ne peuvent pas récupérer l'information inter-modale perdue lors de l'extraction indépendante.

\subsection{Fusion hybride : architectures multi-étages}

La fusion hybride combine les niveaux précoce et tardif en cascade, capitalisant sur les forces complémentaires de chaque approche. Aswin \textit{et al.} \cite{aswin2025deep} démontrent qu'une stratégie hybride (70\% score, 30\% feature) atteint 99,79\% de précision sur NUPT-FPV, surpassant la fusion purement score-level (99,37\%) et purement feature-level. De manière analogue, Al-Askar \textit{et al.} \cite{alaskar2025deep} rapportent un EER de 0,18\% en fusion hybride contre 0,28\% en fusion score-level sur la même base.

Cette supériorité s'explique par la complémentarité informationnelle : la fusion précoce capture les caractéristiques biométriques fines, tandis que la fusion tardive reflète la confiance du classifieur. Plusieurs architectures sophistiquées émergent dans ce paradigme. Le \textit{Multimodal Bottleneck Transformer} (MBT) de Nagrani \textit{et al.} \cite{nagrani2021attention} force l'information inter-modale à transiter par un petit nombre de \textit{bottleneck latents}, obligeant le modèle à condenser l'information la plus pertinente. AuthFormer \cite{yang2024authformer} combine mécanisme d'attention croisée et \textit{Gated Residual Networks} (GRN) pour fusionner dynamiquement visage, empreinte et voix — atteignant 99,73\% de précision avec seulement deux couches d'encodeur.

Le coût computationnel de ces architectures hybrides demeure élevé. La distillation de connaissances parallèle à attention multi-niveaux (MPAD) \cite{lu2025mpad} compresse les réseaux multimodaux complexes via un module de fusion par attention parallèle (PAFM) tout en maintenant les performances. À l'extrême, les réseaux de neurones à spikes (SNN) \cite{shen2025energy} atteignent 95,31\% de précision sur CASIA avec seulement 1,32 mJ de consommation énergétique — environ le centième des méthodes ANN conventionnelles — par une fusion inspirée de la théorie de l'information \textit{bottleneck}.

QME (\textit{Quality-Guided Mixture of Score-Fusion Experts}) \cite{zhu2025qme}, présenté à l'ICCV 2025, propose une stratégie de fusion learnable par \textit{Mixture of Experts} qui s'adapte au bruit capteur, aux occlusions et aux modalités manquantes. AuthFormer et QME représentent l'état de l'art en fusion hybride, mais aucun ne traite nativement les quatre défis simultanément — la quantification d'incertitude, la gestion des modalités manquantes, et l'explicabilité bimodale restent absents.

\subsection{Synthèse comparative}

Le Tableau \ref{tab:comparais_fusion} synthétise les propriétés des trois approches.

\begin{table}[htbp]
\centering
\caption{Comparaison des stratégies de fusion profonde pour la biométrie multimodale.}
\label{tab:comparais_fusion}
\small
\begin{tabular}{>{\raggedright}p{2.8cm}>{\raggedright}p{4.2cm}>{\raggedright}p{4.5cm}>{\raggedright\arraybackslash}p{3.5cm}}
\hline
\textbf{Type} & \textbf{Avantages} & \textbf{Limitations} & \textbf{Référence clé} \\
\hline
\hline
Fusion précoce & Riches interactions croisées ; apprentissage joint de représentations & Propagation de corruption modale ; fragilité à la dégradation & Soleymani \textit{et al.} \cite{soleymani2018generalized} \\
\hline
Fusion tardive & Modularité ; gestion simple des modalités manquantes ; facilement déployable & Perte irréversible d'information inter-modale ; poids globaux fixes & Nandakumar \textit{et al.} \cite{nandakumar2008likelihood} \\
\hline
Fusion hybride & Combine les atouts des deux approches ; précision maximale rapportée & Complexité architecturale élevée ; coût computationnel & Aswin \textit{et al.} \cite{aswin2025deep}, Yang \textit{et al.} \cite{yang2024authformer} \\
\hline
\end{tabular}
\end{table}

\section{Mécanismes d'Attention et Transformers pour la Biométrie}
\label{sec:attention}

\subsection{Self-attention pour la reconnaissance faciale}

L'adoption des \textit{Vision Transformers} (ViT) pour la reconnaissance faciale s'est accélérée après les travaux fondateurs de Zhong \& Deng \cite{zhong2021face}. TransFace \cite{dan2023transface}, présenté à l'ICCV 2023, a identifié les incompatibilités critiques entre l'entraînement ViT et la reconnaissance faciale à grande échelle, proposant une perturbation d'amplitude de patchs dominants (DPAP) et une stratégie de minage d'échantillons durs basée sur l'entropie (EHSM) — TransFace-S surpasse ViT-S de +0,56\% sur IJB-C à TAR@FAR=$10^{-4}$. LVFace \cite{you2025lvface} pousse cette logique plus loin avec une optimisation progressive de clusters (PCO) en trois étapes, atteignant la première place du classement MFR-Ongoing en mars 2025 avec 97,25\% TAR@FAR=$10^{-5}$. ViT-GCT \cite{chettaoui2026vitgct}, soumis à l'ICLR 2026, remplace le token CLS classique par un \textit{Global Context Token} qui interagit avec tous les patchs via self-attention, induisant un focus renforcé sur les landmarks faciaux clés (+1,0\% sur IJB-C à FAR=$10^{-5}$).

TopoFR \cite{dan2024topoFR} exploite l'homologie persistante pour aligner les structures topologiques entre l'espace d'entrée et l'espace latent, surpassant même TransFace sur certains benchmarks avec un simple ResNet50 — suggérant que la stratégie d'entraînement prime parfois sur l'architecture. FaceCoresetNet \cite{shapira2024facecoreset}, présenté à l'AAAI 2024, reformule la reconnaissance d'ensembles faciaux comme un problème de sélection de \textit{coreset} différentiable, utilisant la cross-attention entre un petit ensemble de tokens \textit{core-template} et le template facial complet pour enrichir le descripteur. Ces travaux établissent les transformers comme l'architecture dominante pour la biométrie unimodale.

Pour les empreintes digitales, AFR-Net \cite{grosz2024afrnet} combine des embeddings CNN (ResNet50) et des blocs d'attention transformer (12 couches, 8 têtes), atteignant 99,78\% de rank-1 sur NIST SD14 contre une galerie de 100K — surpassant DeepPrint (99,20\%). Ridgeformer \cite{pandey2025ridgeformer} introduit une cross-attention entre les tokens de deux échantillons d'empreintes pour l'appariement cross-domain (contactless vers contact). IFViT \cite{qiu2024ifvit} propose une représentation de longueur fixe interprétable via un réseau siamois ViT avec cross-attention entre paires d'empreintes, établissant des correspondances pixel-à-pixel.

\subsection{Attention inter-modale : fusion guidée par la qualité}

La modulation de l'attention par la qualité de l'échantillon biométrique s'est imposée comme un impératif opérationnel. Soleymani \textit{et al.} \cite{soleymani2021quality} démontrent qu'une fusion feature-level guidée par des scores de qualité faiblement supervisés surpasse les fusions rank-level et score-level de plus de 30\% en TAR à FAR=$10^{-4}$ sur la base BIOMDATA. Ce résultat est crucial : il établit que la modulation de la qualité doit impérativement opérer au niveau des représentations, et non des scores, pour être informationnellement efficace.

Gnanapraveen \textit{et al.} \cite{gnanapraveen2024cross}, présenté à Odyssey 2024, confirment empiriquement la supériorité de l'attention croisée : sur la reconnaissance audio-visuelle de personnes, l'attention croisée atteint un EER de 2,387 contre 2,412 (self-attention), 2,489 (concaténation de features) et 2,521 (fusion score-level). Cette hiérarchie — cross-attention > self-attention > concaténation > score-level — fournit un argument quantitatif solide pour le positionnement d'UFM-Transformer. Elle suggère également une dualité fonctionnelle essentielle : les poids d'attention servent simultanément à moduler la fusion et à fournir une explication inhérente des décisions, chaque poids révélant quelle région d'une modalité «\,regarde\,» quelle région de l'autre. Cette propriété \textit{dual-use}, exploitée explicitement dans UFM-Transformer, n'a pas été reconnue comme telle dans les travaux existants : Tiong \textit{et al.} \cite{tiong2024flexible} utilisent l'attention pour la fusion mais ne visualisent pas les patterns attentionnels ; Selvarani \& Rani \cite{selvarani2025explainability} analysent les poids d'attention mais via SHAP/LIME \textit{post-hoc}.

\subsection{Les transformers croisés : de la vision multimodale à la biométrie}

L'attention croisée entre modalités biométriques distinctes reste paradoxalement sous-explorée. Tiong \textit{et al.} \cite{tiong2024flexible} proposent MFA-ViT (\textit{Multimodal Fusion Attention}), qui établit l'état de l'art pour la reconnaissance face+pério-oculaire par une attention de fusion multimodale combinée à un \textit{prompt tuning} multimodal — évitant le \textit{fine-tuning} spécifique à chaque modalité. Les \textit{Tiny Transformers} \cite{tiny2026transformers} intègrent un mécanisme de \textit{Quality-Gated Fusion} utilisant un MLP léger pour pondérer dynamiquement les scores de similarité selon la qualité d'entrée (mesure de netteté Laplacienne).

Plusieurs architectures récentes illustrent la richesse du paradigme transformer pour la fusion, sans toutefois combler le gap visé. AuthFormer \cite{yang2024authformer} emploie une attention croisée entre visage, empreinte et voix suivie de réseaux résiduels à gating (GRN) pour un poids total réduit — seulement deux couches d'encodeur suffisent à atteindre 99,73\% de précision sur la base LUTBIO. Les \textit{Tiny Transformers} \cite{tiny2026transformers} proposent une architecture hybride ViT+Swin avec fusion par gating de qualité, atteignant une latence inférieure à 50ms avec moins de 3 millions de paramètres. QME \cite{zhu2025qme} adopte une \textit{Mixture of Experts} guidée par la qualité pour la reconnaissance biométrique \textit{whole-body} (visage+démarche+corps), adaptant dynamiquement la stratégie de fusion au bruit capteur et aux modalités absentes.

Pourtant, une observation d'importance critique émerge de cette revue : aucun travail publié n'implémente une attention croisée inter-modale \textit{directe} entre des patchs faciaux et des patchs d'empreintes digitales au sein d'un espace transformer partagé. Les approches multimodales existantes utilisent systématiquement des encodeurs séparés suivis d'une fusion tardive — au niveau des scores ou par concaténation d'embeddings. MFA-ViT \cite{tiong2024flexible} applique l'attention croisée au couple face+pério-oculaire ; les transformers audio-visuels \cite{gnanapraveen2024cross} traitent le couple voix+visage ; AuthFormer \cite{yang2024authformer} fusionne visage, empreinte et voix mais par attention croisée séquentielle avec encodeurs séparés, non par co-embedding token-level des patchs visage et empreinte dans un espace transformer unifié. Cette absence constitue le premier pilier du gap que UFM-Transformer comble.

\subsection{Gap : absence d'attention croisée visage-empreinte}

L'attention croisée a été appliquée au visage+pério-oculaire \cite{tiong2024flexible}, au visage+voix \cite{gnanapraveen2024cross}, et à l'empreinte+veine de doigt \cite{wang2022finger}. Mais la paire visage-empreinte digitale — pourtant la plus déployée operationnellement — n'a jamais fait l'objet d'une étude dédiée avec attention croisée token-level. UFM-Transformer propose précisément cette architecture inédite.

\section{Robustesse aux Données Dégradées et Modalités Absentes}
\label{sec:robustesse}

\subsection{Reconnaissance faciale robuste : occlusion, flou, bruit}

La reconnaissance faciale sous dégradation a connu des avancées substantielles portées par des fonctions de perte adaptatives à la qualité. AdaFace \cite{kim2022adaface} introduit une marge adaptative fonction de la qualité estimée par la norme du feature vector, améliorant l'état de l'art sur IJB-B, IJB-C, IJB-S et TinyFace. MagFace \cite{meng2021magface} génère des embeddings sensibles à la qualité par une marge angulaire consciente de la magnitude, naturellement exploitée par QMagFace \cite{terhorst2023qmagface} pour des scores de comparaison sensibles à la qualité (98,74\% sur CFP-FP).

L'estimation de qualité faciale (FIQA) a atteint une maturité significative. CR-FIQA \cite{boutros2023crfiqa} apprend la classifiabilité relative d'échantillons en sondeant les observations internes du réseau durant l'entraînement, classant 1er/2ème à l'évaluation NIST FATE Quality. SER-FIQ \cite{terhorst2020serfiq} propose une approche non supervisée exploitant la robustesse stochastique de sous-réseaux aléatoires. GraFIQs \cite{kolf2024grafiqs} introduit une approche complètement \textit{training-free} exploitant les magnitudes de gradient des statistiques de \textit{Batch Normalization}.

Pour l'occlusion, WebFace-OCC \cite{huang2021webface} propose une base de 804\,704 images synthétiquement occluses de 10\,575 sujets ; l'entraînement d'ArcFace sur cette base améliore significativement la robustesse. SlackedFace \cite{low2023slackedface} introduit une marge «\,relâchée\,» pour la reconnaissance basse-résolution, dévalorisant les échantillons durs non reconnaissables — atteignant 80\% de rank-1 sur SCFace D1 étendu. La question de la \textit{cross-quality matching} a été formalisée par la base XQLFW \cite{knoche2021xqlfw}, qui révèle une chute de performance de 25,28\% pour ArcFace comparé à LFW standard, mettant en évidence que les performances sur bases contrôlées ne prédisent pas la robustesse réelle en conditions dégradées.

\subsection{Reconnaissance d'empreintes dégradées : partielles, latentes, sans contact}

La reconnaissance d'empreintes partielles a migré des approches par minuties aux représentations profondes de longueur fixe. PFVNet \cite{he2022pfvnet} apprend une vérification d'empreintes partielles à partir de grandes bases d'appariement, produisant des embeddings invariants à la pose. JIPNet \cite{guan2025jipnet} optimise conjointement la vérification d'identité et l'alignement de pose. Pour les empreintes latentes, LFR-Net \cite{grosz2023lfrnet} fusionne des descripteurs de minuties locaux avec des embeddings globaux AFR-Net via une stratégie de recherche multi-étages — 84,11\% de rank-1 sur NIST SD27 contre une galerie de 100K.

Le passage aux capteurs sans contact post-COVID a intensifié la recherche. C2CL \cite{grosz2022c2cl} atteint un TAR de 96,67--98,30\% à FAR=0,01\% pour l'appariement cross-capteur en combinant pré-traitement (segmentation, enhancement, déroulage) avec appariement hybride minuties+texture. La qualité des empreintes sans contact pose un défi spécifique : MCLFIQ \cite{priesnitz2023mclfiq} montre que NFIQ 2.0 se dégrade sur les données contactless, nécessitant un réentraînement spécifique. L'approche la plus prometteuse pour la généralisation cross-capteur reste l'apprentissage auto-supervisé : Gazi \textit{et al.} \cite{gazi2026minutiae} démontrent qu'un DINOv2-Base (86M paramètres) fine-tuné sur 64\,801 images de 12 bases hétérogènes atteint 5,56\% d'EER en test multi-capteur, contre 26,90\% pour VeriFinger et 41,95\% pour SourceAFIS — soit une amélioration relative de 4,8× sur les systèmes commerciaux.

\subsection{Estimation de qualité : NFIQ 2.0 et réseaux profonds}

NFIQ 2.0 \cite{tabassi2021nfiq}, standard ISO/IEC 29794-1, fournit une mesure de qualité dans [0,100] via une forêt aléatoire entraînée sur 74 caractéristiques de qualité. Pourtant, Priesnitz \textit{et al.} \cite{priesnitz2025deep} démontrent à l'BIOSIG 2025 que des CNN fine-tunés (VGG16) surpassent NFIQ 2.3 de 12,29\% en performance prédictive mesurée par courbes EDC. MCLFIQ \cite{priesnitz2023mclfiq} adapte NFIQ 2.0 aux empreintes sans contact mobiles, surpassant NFIQ 2.2 sur toutes les bases testées. Pour le visage, les normes ISO/IEC 29794-5 définissent des mesures de qualité héuristiques tandis que les méthodes d'UQ profonde (CR-FIQA, SER-FIQ) apprennent l'incertitude directement des données. Le lien entre ces deux approches — qualité heuristique standardisée et incertitude apprise — demeure inexploré et sera investigué dans ce mémoire.

\subsection{Gestion des modalités manquantes : un retard critique}

La gestion des modalités manquantes constitue le retard le plus flagrant de la biométrie par rapport aux domaines voisins. En imagerie médicale, des solutions sophistiquées existent : SMIL \cite{ma2021smil} gère jusqu'à 90\% de données manquantes par méta-régularisation bayésienne ; mmFormer \cite{zhang2022mmformer} traite toute sous-ensemble de modalités IRM ; le \textit{prompt tuning} avec modalités manquantes \cite{lee2023multimodal} nécessite moins de 1\% de paramètres apprenables ; les \textit{Cross-Modal Proxy Tokens} (CMPT) \cite{reza2025cmpt} approximent le token de classe d'une modalité absente à partir des modalités disponibles.

En biométrie multimodale, la situation est radicalement différente. Sur 17 méthodes de gestion de modalités manquantes recensées, seules 3 sont spécifiquement biométriques — et toutes utilisent des approches non profondes (\textit{fallback} score-level ou fusion décisionnelle) \cite{shen2025energy, elbousty2025multimodal, korse2026context}. Shen \textit{et al.} \cite{shen2025energy} introduisent le \textit{Multimodal Adaptive Dropout} (MAD) pour l'équilibrage modal lors de l'entraînement, mais cible l'\textit{imbalance} plutôt que l'absence à l'inférence. El-Bousty \textit{et al.} \cite{elbousty2025multimodal} implémentent des stratégies de repli explicites pour l'authentification mono- et bi-modale. Korse \textit{et al.} \cite{korse2026context} proposent une fusion décisionnelle avec détection de contradiction et gating de fiabilité pour le contrôle d'accès. Aucun de ces travaux n'adopte les méthodes d'avant-garde du machine learning général (proxy tokens, prompt tuning, LoRA adaptation) pour compenser les modalités manquantes au niveau représentationnel. Ma \textit{et al.} \cite{ma2022multimodal} ont établi de manière frappante que les transformers multimodaux ne sont \textit{pas} naturellement robustes aux modalités manquantes — ils peuvent même performer \textit{moins bien} que des modèles unimodaux. Neverova \textit{et al.} \cite{neverova2016moddrop} avaient introduit le \textit{ModDrop} (abandon aléatoire de canaux modaux durant l'entraînement), technique fondamentale mais sous-utilisée en biométrie. Ce retard de plus de cinq ans par rapport à l'imagerie médicale constitue le deuxième pilier du gap identifié.

\section{Quantification d'Incertitude et Calibration des Scores}
\label{sec:incertitude}

\subsection{Incertitude en apprentissage profond : épistémique vs aléatoire}

La distinction entre incertitude épistémique (liée au modèle, réductible avec plus de données) et incertitude aléatoire (liée aux données, irréductible), formalisée par Kendall \& Gal \cite{kendall2017uncertainties}, structure les approches contemporaines. En biométrie, cette dichotomie revêt une importance opérationnelle : l'incertitude épistémique signale un manque de connaissance du système (nouvelle identité, biais de distribution), tandis que l'incertitude aléatoire reflète la qualité dégradée de l'échantillon (flou, occlusion, bruit capteur). Cette séparation motive la modélisation probabiliste des embeddings dans UFM-Transformer, où la variance du modèle capture l'incertitude épistémique et la variance des features capture l'incertitude aléatoire.

Trusted Multi-View Classification (TMC) \cite{han2022trusted} illustre cette convergence en appliquant l'apprentissage profond évidentiel avec fusion dynamique de Dempster-Shafer pour la classification multi-vue. Bien que non spécifique à la biométrie, ce cadre fournit une méthode principielle pour fusionner des preuves incertaines de vues multiples avec quantification explicite de l'incertitude — une philosophie directement transposable à la fusion visage-empreinte.

\subsection{Monte-Carlo Dropout et Deep Ensembles}

Joshi \textit{et al.} \cite{joshi2022estimating} ont appliqué le MC-Dropout à l'amélioration d'empreintes via MU-GAN et DU-GAN, démontrant que l'incertitude modèle est élevée aux pixels incorrectement prédits tandis que l'incertitude données est élevée sur les régions bruitées. Cependant, le MC-Dropout multiplie le temps d'inférence par dix (49,32ms contre 5,13ms pour 10 échantillons). Les \textit{Deep Ensembles}, pourtant \textit{gold standard} en UQ selon Fort \textit{et al.} \cite{fort2019deep}, demeurent sous-utilisés en biométrie — la plupart des travaux se limitent à des approches single-model. HUE-Net \cite{huenet2025human} propose une architecture combinant MC-Dropout avec rétro-propagation bayésienne pour quantifier les deux types d'incertitude en reconnaissance faciale sous conditions adverses, mais reste uni-modal.

\subsection{Embeddings probabilistes : PFE, DUL et au-delà}

Les \textit{Probabilistic Face Embeddings} (PFE) de Shi \& Jain \cite{shi2019pfe} représentent chaque image faciale comme une distribution gaussienne $\mathcal{N}(\mathbf{z}; \boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ dans l'espace latent, où $\boldsymbol{\mu}$ encode l'identité et $\boldsymbol{\sigma}^2$ quantifie l'incertitude. Le \textit{Mutual Likelihood Score} (MLS) remplace la similarité cosinus pour l'appariement, améliorant les performances et fournissant des «\,bons indicateurs de la précision d'appariement potentielle\,».

Chang \textit{et al.} \cite{chang2020dul} étendent le PFE par l'apprentissage conjoint de la représentation et de l'incertitude (\textit{Data Uncertainty Learning}), atteignant 90,23\% de TPR@FPR=$10^{-5}$ sur IJB-C contre 89,64\% (PFE) et 83,06\% (baseline). La variance apprise corrèle avec la dégradation de qualité de l'image et sert d'«\,indicateur de risque\,». Chen \textit{et al.} \cite{chen2022dual} proposent une modélisation par double gaussienne, séparant les échantillons faciles des échantillons difficiles (poses extrêmes, bruit important) par deux distributions distinctes, corrélant l'incertitude avec plusieurs attributs simultanément.

HolUE \cite{erlygin2024holistic} combine l'incertitude \textit{gallery-aware} (GalUE) avec des estimations de qualité via un modèle bayésien, surpassant PFE, SCF et SF pour la reconnaissance ouverte contrôlée en risque sur IJB-C. LAM (\textit{Laplace Approximation Metric learning}) \cite{brack2023bayesian} emploie l'approximation de Laplace sur les poids du réseau plutôt que l'amortissement neuronal, produisant des incertitudes bien calibrées avec un ECE inférieur aux méthodes MC-Dropout et Deep Ensembles.

\subsection{Calibration : ECE, MCE, temperature scaling}

La calibration des scores de similarité biométriques reçoit une attention renouvelée. L'\textit{Expected Calibration Error} (ECE) et le \textit{Maximum Calibration Error} (MCE), empruntés à la littérature de classification, mesurent l'écart entre la confiance prédite et la précision effective. Conti \textit{et al.} \cite{conti2024assessing}, présenté à l'ICLR 2024, démontrent que le bootstrap naïf échoue pour le scoring de similarité car les FAR/FRR sont des U-statistiques ; leur bootstrap recentré fournit des bandes de confiance valides autour des courbes ROC et des métriques d'équité. Le \textit{temperature scaling} et le \textit{Platt scaling}, standards en classification, n'ont pas été systématiquement validés pour les distributions de scores de similarité biométriques — où les sorties sont des similarités cosinus ou des distances euclidiennes plutôt que des probabilités softmax. La prédiction conforme (\textit{Conformal Prediction}) \cite{balomenos2018automatic}, offrant des garanties de couverture à échantillon fini, demeure inexplorée pour la fusion biométrique multimodale. Ces lacunes motiveront la procédure de calibration développée au Chapitre \ref{chap:methode}.

\subsection{Gap : aucun système biométrique multimodal avec quantification d'incertitude}

Malgré la richesse des travaux sur l'incertitude unimodale et la fusion multimodale, leur intersection demeure quasiment vide. Seuls deux travaux émergent : la fusion Dempster-Shafer de Nguyen \textit{et al.} \cite{nguyen2015score} modélise explicitement trois masses (\textit{genuine}, \textit{impostor}, \textit{incertain}) au niveau des scores ; et la fusion context-aware de Zibert \textit{et al.} \cite{zibert2026context} détecte les contradictions entre modalités biométriques au niveau décisionnel. Aucun système ne propage l'incertitude de chaque modalité à travers le processus de fusion pour produire des scores de confiance multimodaux calibrés avec option de rejet. Ce constat constitue le troisième pilier du gap.

\section{Explicabilité des Systèmes Biométriques Multimodaux}
\label{sec:explicabilite}

\subsection{Grad-CAM et dérivés pour la reconnaissance faciale}

Les méthodes de visualisation par gradient, dominées par Grad-CAM et Grad-CAM++ \cite{selvaraju2017grad}, ont été largement adoptées pour expliquer les décisions de reconnaissance faciale. Pourtant, Lu \textit{et al.} \cite{lu2024corrRISE}, présenté au WACV 2024, établissent que ces méthodes produisent des cartes de saillance moins stables et moins significatives que les approches par perturbation. Leur méthode CorrRISE (\textit{Correlation-based Randomized Input Sampling for Explanation}) surpasse systématiquement Grad-CAM, Grad-CAM++, LIME, RISE, MinPlus et xFace sur les benchmarks de vérification faciale — avec un taux de suppression (Deletion) de 23,29\% contre 50,65\% pour Grad-CAM sur LFW.

La survey de Neto \textit{et al.} \cite{neto2022explainable} établit une taxonomie systématique des techniques XAI pour la biométrie, structurant le domaine selon une «\,échelle XAI\,» à trois niveaux : visualisation par gradient, méthodes post-hoc apprenables, et conception ante-hoc interprétable. Le champ connaît une tension méthodologique structurante : les méthodes par gradient (Grad-CAM, xSSAB) privilégient l'efficacité computationnelle tandis que les méthodes par perturbation (CorrRISE, RISE) offrent des explications de meilleure qualité mais à un coût bien supérieur. Sur le benchmark Patch-LFW \cite{huber2024xssab}, CorrRISE atteint un taux d'insertion (Insertion) de 85,70\% contre 65,00\% pour Grad-CAM, établissant une hiérarchie de qualité claire. Cependant, cette qualité s'acquiert au prix d'une latence incompatible avec le temps réel — ce qui motive l'approche d'explicabilité \textit{ante-hoc} d'UFM-Transformer, où les poids d'attention sont des sous-produits gratuits de la fusion.

Huber \textit{et al.} \cite{huber2024xssab} proposent xSSAB (\textit{similarity score argument backpropagation}), une approche \textit{white-box} \textit{training-free} qui rétro-propage le score de similarité cosinus à travers un réseau siamois pour indiquer les zones similaires et dissimilaires. Lin \textit{et al.} \cite{lin2021xcos} développent xCos, un module apprenable fournissant des explications spatiales pondérées par l'attention. Rocha \textit{et al.} \cite{rocha2025protosiam} introduisent ProtoSiamese et MaskSiamese, des réseaux siamois intrinsèquement interprétables par comparaison explicite patch-à-patch.

\subsection{Cartes d'attention pour l'importance des minuties d'empreinte}

Tan \& Kumar \cite{tan2021minutiae} proposent un réseau d'attention aux minuties avec \textit{reciprocal distance loss}, générant des cartes de vraisemblance des minuties servant de cartes d'attention pour guider le réseau vers les zones distordues. Découverte plus récente et d'importance capitale : Gazi \textit{et al.} \cite{gazi2026minutiae} démontrent qu'un ViT auto-supervisé DINOv2 pour l'empreinte digitale apprend spontanément des cartes d'attention qui s'alignent avec les régions riches en minuties (IoU = 0,41 $\pm$ 0,07), bien qu'aucune supervision minutie n'ait été fournie. Les tokens d'attention se concentrent systématiquement sur les transitions de crêtes, les régions à forte courbure et les motifs ressemblant à des bifurcations — éléments structuraux discriminants par excellence. Ce phénomène — l'explicabilité «\,gratuite\,» des transformers auto-supervisés — constitue un argument fort pour l'adoption de backbones ViT dans UFM-Transformer, car il promet que les poids d'attention croisée entre visage et empreinte seront intrinsèquement interprétables, révélant quelles régions faciales correspondent à quelles zones d'empreinte pour l'appariement identitaire. Ce phénomène — l'explicabilité «\,gratuite\,» des transformers auto-supervisés — constitue un argument fort pour l'adoption de backbones ViT dans UFM-Transformer.

\subsection{Explicabilité multimodale : SHAP, LIME, attention cross-modale}

Selvarani \& Rani \cite{selvarani2025explainability} intègrent pour la première fois SHAP et LIME dans un pipeline d'authentification multimodale (visage, empreinte, paume), quantifiant les contributions modales : visage 48\%, empreinte 31\%, paume 21\%. Leur réseau de fusion cross-modale par attention alloue dynamiquement l'importance à chaque modalité. Cependant, SHAP et LIME fournissent une explicabilité \textit{post-hoc} : ils expliquent une décision déjà prise sans que le mécanisme explicatif ne soit intégré à l'architecture. Cette séparation entre raisonnement et explication limite la confiance opérationnelle, d'autant que les méthodes \textit{post-hoc} peuvent produire des explications inconsistents avec le comportement réel du modèle \cite{neto2022explainable}.

La taxonomie de Neto \textit{et al.} \cite{neto2022explainable} établit une «\,échelle XAI\,» à trois niveaux : (1) visualisation par gradient, (2) méthodes post-hoc apprenables, (3) conception ante-hoc interprétable. Les réseaux d'attention graphe (GAT) de Harini \textit{et al.} \cite{harini2025graph} représentent un pas vers l'ante-hoc en traitant les modalités biométriques comme des nœuds d'un graphe pleinement connecté, les poids d'attention inter-modaux fournissant une explicabilité inhérente. ProtoSiamese et MaskSiamese \cite{rocha2025protosiam} vont plus loin en proposant des réseaux siamois intrinsèquement interprétables par comparaison explicite patch-à-patch, MaskSiamese surpassant même le réseau siamois boîte-noire tout en fournissant des explications significatives.

\subsection{Gap : absence d'explication visuelle bimodale unifiée}

Aucun travail publié ne fournit d'explications visuelles unifiées mettant simultanément en évidence les régions faciales \textit{et} les points minuties d'empreinte ayant contribué à une décision bimodale. Les travaux existants quantifient les contributions au niveau modalité (SHAP) ou génèrent des cartes de saillance unimodales (CorrRISE pour le visage, attention aux minuties pour l'empreinte), mais jamais les deux simultanément dans une visualisation fusionnée. Ce quatrième gap est représenté schématiquement à la Figure \ref{fig:gap_explicabilite}.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\textwidth]{fig_gap_explicabilite.pdf}
\caption{Illustration schématique du gap d'explicabilité bimodale : (a) explication unimodale visage par CorrRISE, (b) explication unimodale empreinte par attention aux minuties, (c) l'explication bimodale unifiée que UFM-Transformer fournit via les poids d'attention croisée. Aucun travail existant ne produit (c).}
\label{fig:gap_explicabilite}
\end{figure}

\section{Identification du Gap et Positionnement}
\label{sec:gap}

\subsection{Synthèse des gaps par dimension}

Le Tableau \ref{tab:gaps} synthétise les lacunes identifiées pour chaque dimension, avec les travaux les plus proches et leurs limites.

\begin{table}[htbp]
\centering
\caption{Synthèse des gaps scientifiques par dimension de l'état de l'art.}
\label{tab:gaps}
\small
\begin{tabular}{>{\raggedright}p{2.5cm}>{\raggedright}p{5.5cm}>{\raggedright}p{4.5cm}>{\raggedright\arraybackslash}p{3.0cm}}
\hline
\textbf{Dimension} & \textbf{Gap identifié} & \textbf{Travaux les plus proches} & \textbf{Limite de ces travaux} \\
\hline
\hline
Fusion & Aucune attention croisée token-level entre patchs visage et patchs empreinte & Tiong \textit{et al.} \cite{tiong2024flexible} (face+pério-oculaire) ; Gnanapraveen \textit{et al.} \cite{gnanapraveen2024cross} (audio-visuel) & Modalités différentes ; pas de paire visage-empreinte \\
\hline
Robustesse & Aucune gestion native de modalités manquantes pour visage+empreinte & SMIL \cite{ma2021smil}, CMPT \cite{reza2025cmpt}, Lee \textit{et al.} \cite{lee2023multimodal} & Domaine imagerie médicale ; jamais appliqué à la biométrie \\
\hline
Incertitude & Aucun système ne propage l'incertitude à travers la fusion multimodale & PFE/DUL \cite{shi2019pfe, chang2020dul} (unimodal) ; Dempster-Shafer \cite{nguyen2015score} & Uniquement au niveau des scores ; pas de propagation feature-level \\
\hline
Explicabilité & Aucune explication visuelle bimodale unifiée & CorrRISE \cite{lu2024corrRISE} (visage) ; attention minuties \cite{tan2021minutiae} (empreinte) & Explications unimodales séparées \\
\hline
\end{tabular}
\end{table}

\subsection{UFM-Transformer face aux travaux existants}

Le Tableau \ref{tab:comparatif_ufm} positionne UFM-Transformer contre six représentants de l'état de l'art les plus proches, selon huit critères de comparaison.

\begin{table}[htbp]
\centering
\caption{Comparatif d'UFM-Transformer avec les travaux les plus proches de l'état de l'art selon huit dimensions. \checkmark = satisfait pleinement, $(\circ)$ = satisfait partiellement, $\times$ = non satisfait.}
\label{tab:comparatif_ufm}
\small
\begin{tabular}{lccccccc}
\hline
\textbf{Critère} & \textbf{\cite{yang2024authformer}} & \textbf{\cite{zhu2025qme}} & \textbf{\cite{tiong2024flexible}} & \textbf{\cite{soleymani2021quality}} & \textbf{\cite{shi2019pfe}} & \textbf{\cite{lu2024corrRISE}} & \textbf{UFM-T} \\
\hline
\hline
Fusion feature-level & \checkmark & $\times$ & \checkmark & \checkmark & $\times$ & $\times$ & \checkmark \\
Attention croisée modale & \checkmark & $\times$ & \checkmark & $\times$ & $\times$ & $\times$ & \checkmark \\
Modulation par qualité & $(\circ)$ & \checkmark & $(\circ)$ & \checkmark & $\times$ & $\times$ & \checkmark \\
Modalités manquantes & $(\circ)$ & $(\circ)$ & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark \\
Quantif. incertitude & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark & $\times$ & \checkmark \\
Explicabilité visuelle & $\times$ & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark & \checkmark \\
Explicabilité bimodale & $\times$ & $\times$ & $\times$ & $\times$ & $\times$ & $\times$ & \checkmark \\
Calibration probabiliste & $\times$ & $\times$ & $\times$ & $\times$ & $(\circ)$ & $\times$ & \checkmark \\
\hline
\end{tabular}
\end{table}

\subsection{Le gap central}

L'analyse systématique de plus de 150 articles couvrant dix dimensions de recherche révèle un phénomène structurel : la littérature biométrique s'est fragmentée en «\,îlots architecturaux\,» non communicants \cite{liang2024deep}. Les chercheurs en fusion précoce optimisent la concaténation de features ; ceux en fusion tardive affinent les règles de combinaison ; les communautés d'incertitude et d'explicabilité évoluent séparément. À travers ces 150 travaux, aucun ne satisfait plus de deux des quatre exigences simultanément.

AuthFormer \cite{yang2024authformer} atteint l'attention croisée avec adaptation modale mais ignore l'incertitude, les modalités manquantes et l'explicabilité bimodale. QME \cite{zhu2025qme} gère la qualité et les modalités manquantes mais opère au niveau des scores et ne quantifie pas l'incertitude. PFE/DUL \cite{shi2019pfe, chang2020dul} fournissent des embeddings probabilistes calibrés mais pour la reconnaissance faciale unimodale uniquement. CorrRISE \cite{lu2024corrRISE} offre une explicabilité visuelle sophistiquée mais pour le visage seul.

\textbf{Le gap central} que ce mémoire comble s'énonce ainsi : \textit{aucun système biométrique multimodal existant ne traite simultanément les quatre défis} — (i) attention croisée visage-empreinte avec modulation par qualité au niveau token, (ii) gestion native des modalités manquantes sans modification architecturale à l'inférence, (iii) propagation de l'incertitude à travers la fusion pour produire des scores calibrés avec option de rejet, et (iv) explication visuelle bimodale unifiée. UFM-Transformer propose une architecture originale qui intègre ces quatre capacités dans un seul framework différentiable, rendant chaque composante — les poids d'attention, les estimations d'incertitude, les cartes d'explication — des sous-produits naturels d'un même mécanisme d'attention croisée probabiliste.

Cette unification architecturale ne se contente pas de coexister quatre techniques séparées : elle les fait interagir de manière synergique. Les poids d'attention croisée modulent la fusion tout en fournissant l'explicabilité ; les variances de features, propagées par la loi de la variance totale, quantifient l'incertitude tout en informant la modulation par qualité ; le \textit{modality dropout} durant l'entraînement prépare l'architecture à l'absence de modalités tout en générant des données de calibration naturelles. Cette cohérence interne distingue fondamentalement UFM-Transformer d'une simple juxtaposition de modules. Le chapitre suivant développe formellement cette architecture, ses équations de propagation, et sa procédure d'entraînement.


