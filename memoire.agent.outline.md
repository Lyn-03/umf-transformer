# Deep Learning Approaches for Multimodal Biometrics: Fusion Uncertainty-aware de Visage et d'Empreinte Digitale par Transformer avec Gestion des Modalités Manquantes

## Résumé (~250 mots en français)
### Résumé de la recherche
#### La fusion biométrique multimodale visage-empreinte digitale souffre d'un alignement sous-optimal des représentations sous dégradation, de l'absence de gestion native des modalités manquantes et de scores déterministes sans intervalle de confiance
#### Nous proposons UFM-Transformer, un transformer à attention croisée modulée par la qualité, avec tokens apprenables pour modalités manquantes, estimation d'incertitude par Monte-Carlo Dropout, et explicabilité bimodale Grad-CAM
#### Résultats sur [DATASET] avec 100+ sujets : réduction de l'EER de \textcolor{red}{\textbf{XX\%}} par rapport à la meilleure baseline

## Abstract (~250 words in English)
### Research Summary
#### Same content as résumé but in English

## Introduction Générale (~2000 mots)
### Contexte et Enjeux
#### 1.1.1 La biométrie multimodale : pourquoi la fusion visage + empreinte digitale ? Traits complémentaires, modes de défaillance orthogonaux
#### 1.1.2 Le paradigme deep learning en biométrie : des CNNs aux Vision Transformers
#### 1.1.3 La vérification d'identité en scénario ouvert : défis spécifiques
### Problématique et Gaps Identifiés
#### 1.1.4 Quatre défis interconnectés : alignement sous dégradation, modalités manquantes, scores sans incertitude, absence d'explicabilité bimodale
#### 1.1.5 Le "problème des îlots modaux" : chaque défi traité isolément dans la littérature
### Contributions de ce Mémoire
#### 1.1.6 Contribution 1 : UFM-Transformer, première architecture unifiée abordant les quatre défis simultanément
#### 1.1.7 Contribution 2 : Transformer à attention croisée modulée par la qualité avec tokens pour modalités manquantes
#### 1.1.8 Contribution 3 : Scores de similarité calibrés avec intervalle de confiance par Monte-Carlo Dropout
#### 1.1.9 Contribution 4 : Grad-CAM bimodal pour l'explicabilité des décisions de vérification
#### 1.1.10 Positionnement par rapport à l'état de l'art : tableau comparatif
### Organisation du Mémoire
#### 1.1.11 Structure des chapitres et fil directeur

## 1. État de l'Art (~5000 mots, 3 tableaux, 1 figure)
### 1.1 Architectures de Fusion Profonde pour la Biométrie Multimodale
#### 1.1.1 Fusion précoce (early fusion) : concatenation, espaces d'embedding joints — avantages et limitations sous dégradation
#### 1.1.2 Fusion tardive (late fusion) : combinaison au niveau des scores — perte d'information inter-modale
#### 1.1.3 Fusion hybride : architectures multi-étages — coût computationnel et complexité
#### 1.1.4 Synthèse comparative des trois approches (Tableau 1.1)
### 1.2 Mécanismes d'Attention et Transformers pour la Biométrie
#### 1.2.1 Self-attention pour la reconnaissance faciale : TransFace, LVFace, adaptations des ViT
#### 1.2.2 Attention inter-modale : fusion guidée par la qualité, attention bottleneck
#### 1.2.3 Les transformers croisés : de la vision multimodale à la biométrie
#### 1.2.4 Gap : absence d'attention croisée visage-empreinte pour la vérification
### 1.3 Robustesse aux Données Dégradées et Modalités Absentes
#### 1.3.1 Reconnaissance faciale robuste : occlusion, flou, bruit — MagFace, AdaFace, CR-FIQA
#### 1.3.2 Reconnaissance d'empreintes dégradées : empreintes partielles, latentes, sans contact
#### 1.3.3 Estimation de qualité : NFIQ 2.0, mesures de flou, réseaux de qualité profonds
#### 1.3.4 Gestion des modalités manquantes : dropout, tokens apprenables, hallucination — retard de 5 ans par rapport à l'imagerie médicale
### 1.4 Quantification d'Incertitude et Calibration des Scores
#### 1.4.1 Incertitude en apprentissage profond : épistémique vs aléatoire
#### 1.4.2 Monte-Carlo Dropout et Deep Ensembles pour la biométrie
#### 1.4.3 Embeddings probabilistes : PFE, DUL pour la reconnaissance faciale
#### 1.4.4 Calibration : ECE, MCE, temperature scaling
#### 1.4.5 Gap : aucun système biométrique multimodal avec quantification d'incertitude
### 1.5 Explicabilité des Systèmes Biométriques Multimodaux
#### 1.5.1 Grad-CAM et dérivés pour la reconnaissance faciale
#### 1.5.2 Cartes d'attention pour l'importance des minuties d'empreinte
#### 1.5.3 Explicabilité multimodale : SHAP, LIME, attention cross-modale
#### 1.5.4 Gap : absence d'explication visuelle bimodale unifiée
### 1.6 Identification du Gap et Positionnement
#### 1.6.1 Tableau synthétique des gaps par dimension (Tableau 1.2)
#### 1.6.2 Tableau comparatif : UFM-Transformer vs travaux existants (Tableau 1.3)
#### 1.6.3 Le gap central : aucun travail existant ne traite simultanément les quatre défis

## 2. Le Modèle UFM-Transformer (~5000 mots, 2 figures, 2 algorithmes)
### 2.1 Formulation du Problème
#### 2.1.1 Notations et définitions formelles
#### 2.1.2 Objectif de vérification ouverte avec qualité variable et modalités partielles
### 2.2 Vue d'Ensemble de l'Architecture
#### 2.2.1 Pipeline global : encodeurs → projecteurs → transformer cross-modal → têtes de décision
#### 2.2.2 Figure : diagramme d'architecture complet avec flux de données
### 2.3 Encodeurs Spécifiques par Modalité
#### 2.3.1 Encodeur visage : EfficientNet-B2, pré-entraînement ImageNet, extraction de cartes de caractéristiques
#### 2.3.2 Encodeur empreinte : CNN résiduel personnalisé pour les motifs de crêtes
#### 2.3.3 Estimateur de qualité : CNN léger par modalité, score dans [0,1]
### 2.4 Projecteur Commun et Gestion des Modalités Manquantes
#### 2.4.1 Projecteurs denses avec normalisation L2 : projection sur l'hypersphère unité
#### 2.4.2 Token apprenable de remplacement : $\mathbf{t}_{miss} \in \mathbb{R}^{256}$ paramètre entraînable
#### 2.4.3 Mécanisme de remplacement : masque booléen et encodage positionnel
### 2.5 Module de Fusion par Attention Croisée Modulée par la Qualité
#### 2.5.1 Attention croisée bidirectionnelle : requêtes visage → clés/valeurs empreinte, et inversement
#### 2.5.2 Modulation par qualité : scores d'attention pondérés par $q_f \cdot q_p$
#### 2.5.3 Masquage pour modalités absentes : attention nulle vers/depuis modalités manquantes
#### 2.5.4 Architecture détaillée : 4 couches, 8 têtes, 256 dimensions
#### 2.5.5 Algorithme : pseudo-code du forward pass
### 2.6 Tête de Similarité et Tête d'Incertitude
#### 2.6.1 Similarité cosinus avec marge additive ArcFace ($m=0.5$, $s=30$)
#### 2.6.2 Tête d'incertitude : 5 passes avant avec dropout actif
#### 2.6.3 Décomposition aléatoire vs épistémique
#### 2.6.4 Intervalle de confiance et option de rejet
### 2.7 Mécanismes d'Explicabilité
#### 2.7.1 Visualisation des cartes d'attention cross-modale
#### 2.7.2 Grad-CAM bimodal : régions faciales et minuties d'empreinte
#### 2.7.3 Analyse des cas d'échec par attention
### 2.8 Fonction de Perte et Stratégie d'Entraînement
#### 2.8.1 Perte composite : triplet + ArcFace + régularisation d'incertitude
#### 2.8.2 Phase 1 : pré-entraînement unimodal (50 epochs, ArcFace seul)
#### 2.8.3 Phase 2 : fine-tuning joint (100 epochs, dropout modalités 30%, perte complète)
#### 2.8.4 Optimisation : AdamW, cosine annealing, gradient clipping
#### 2.8.5 Algorithme : boucle d'entraînement complète

## 3. Expérimentations et Résultats (~5000 mots, 4 tableaux, 4 figures)
### 3.1 Protocole Expérimental
#### 3.1.1 Dataset : Multimodal Biometrics Dataset (Kaggle) — 100+ sujets, 5+ échantillons/modalité
#### 3.1.2 Prétraitement et split sujet-disjoint 70/15/15
#### 3.1.3 Augmentations et simulation de dégradations
#### 3.1.4 Environnement : PyTorch 2.0, GPU NVIDIA, entraînement mixed-precision
### 3.2 Métriques d'Évaluation
#### 3.2.1 EER, TAR@FAR, AUC, FNMR, FMR — définitions
#### 3.2.2 Métriques de calibration : ECE, MCE
#### 3.2.3 Métriques d'efficacité : paramètres, FLOPS, temps d'inférence
### 3.3 Résultats Principaux
#### 3.3.1 Comparaison avec 5 baselines : EER, TAR@0.1%FAR, TAR@1%FAR, AUC (Tableau 3.1)
#### 3.3.2 Analyse des courbes ROC et DET (Figure 3.1)
#### 3.3.3 Distribution des scores : violin plots (Figure 3.2)
### 3.4 Analyse de Robustesse
#### 3.4.1 Évaluation sous 10 conditions de dégradation (Tableau 3.2)
#### 3.4.2 Performance en mode unimodal forcé (visage seul, empreinte seule)
#### 3.4.3 Analyse de dégradation gracieuse avec modalités manquantes
### 3.5 Calibration de l'Incertitude
#### 3.5.1 Diagramme de fiabilité : confiance vs accuracy (Figure 3.3)
#### 3.5.2 ECE et MCE comparés aux baselines (Tableau 3.3)
#### 3.5.3 Performances avec option de rejet (Figure 3.4)
#### 3.5.4 Histogrammes d'incertitude : prédictions correctes vs incorrectes
### 3.6 Étude d'Ablation
#### 3.6.1 Ablation par composant : 5 configurations testées (Tableau 3.4)
#### 3.6.2 Contribution de la modulation par qualité
#### 3.6.3 Contribution du token de modalité manquante
#### 3.6.4 Contribution de la tête d'incertitude
### 3.7 Résultats d'Explicabilité
#### 3.7.1 Heatmaps d'attention cross-modale pour 10 identités tests
#### 3.7.2 Grad-CAM bimodal : les régions faciales et minuties déterminantes
#### 3.7.3 Analyse des cas d'échec : quand et pourquoi le modèle échoue
### 3.8 Efficacité Computationnelle
#### 3.8.1 Nombre de paramètres : UFM-Transformer vs baselines
#### 3.8.2 Temps d'inférence par paire
#### 3.8.3 Discussion faisabilité déploiement temps réel

## 4. Discussion et Perspectives (~2000 mots)
### 4.1 Interprétation des Résultats
#### 4.1.1 Pourquoi UFM-Transformer surpasse les baselines : rôle de chaque composant
#### 4.1.2 L'incertitude comme indicateur fiable de confiance
#### 4.1.3 L'explicabilité comme outil de diagnostic et de confiance utilisateur
### 4.2 Limites
#### 4.2.1 Échelle du dataset : 100+ sujets, besoin de milliers pour l'industrie
#### 4.2.2 Modalités limitées : visage et empreinte uniquement
#### 4.2.3 Coût computationnel du Monte-Carlo Dropout (5 passes)
### 4.3 Perspectives
#### 4.3.1 Extension à d'autres modalités : iris, voix, paume
#### 4.3.2 Distillation de connaissances pour déploiement edge
#### 4.3.3 Apprentissage fédéré pour la biométrie préservant la confidentialité
#### 4.3.4 Prédiction conforme pour des garanties théoriques de calibration

## Conclusion Générale (~1000 mots)
### Synthèse des Contributions
#### C.1.1 UFM-Transformer : architecture unifiée pour la vérification biométrique robuste
#### C.1.2 Résultats clés : EER \textcolor{red}{\textbf{XX\%}}, TAR@0.1\%FAR \textcolor{red}{\textbf{XX\%}}, ECE \textcolor{red}{\textbf{XX\%}}
#### C.1.3 Originalité : première combinaison d'attention qualité-modulée, modalités manquantes, incertitude calibrée, explicabilité bimodale
### Bilan et Ouverture
#### C.1.4 Vers des systèmes biométriques autonomes, confiants et interprétables

# Références
## memoire_outline_references_raw.md
- **Type**: Collection de références
- **Path**: /mnt/agents/output/memoire_outline_references_raw.md

# Annexes
## A. Code Source et Reproductibilité
## B. Détail des Hyperparamètres
## C. Résultats Complémentaires
