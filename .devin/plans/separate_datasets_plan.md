# Plan: Adapter le pipeline pour datasets séparés (CASIA-WebFace + SOCOFing)

## Contexte
- **CASIA-WebFace**: ~10,432 sujets dans `data/casia-webface-extracted/00000/` à `data/casia-webface-extracted/10431/`, chaque dossier contient des images face (000.jpg, 001.jpg, ...)
- **SOCOFing**: 600 sujets, 6,000 empreintes dans `data/SOCOFing/Real/`, nommage `{ID}__{M/F}_Left/Right_{finger}_finger.BMP`
- Les deux datasets n'ont **pas d'identités partagées** → Phase 1 unimodale séparée, Phase 2 avec paires synthétiques

## Étapes

### 1. data_loader.py — Nouvelles classes de dataset
- **`UnimodalFaceDataset`**: parcourt `data/casia-webface-extracted/`, chaque sous-dossier = un sujet, images = échantillons. Retourne `(face_tensor, subject_id, quality)`.
- **`UnimodalFingerprintDataset`**: parse les fichiers SOCOFing `{ID}__*.BMP`, extrait l'ID sujet. Retourne `(fingerprint_tensor, subject_id, quality)`.
- **`SyntheticPairDataset`**: pour Phase 2. Prend un `UnimodalFaceDataset` et un `UnimodalFingerprintDataset`. À chaque `__getitem__`, échantillonne aléatoirement un face et un fingerprint (sujets différents). Retourne `(face, fingerprint, face_subject_id, fp_subject_id, face_quality, fp_quality)`.
- **`get_dataloaders_separate()`**: nouvelle factory qui crée les dataloaders Phase 1 (face + fingerprint séparés) et Phase 2 (paires synthétiques).

### 2. train.py — TrainConfig
- Ajouter `face_path: str` et `fingerprint_path: str` au `TrainConfig`
- Garder `dataset_path` pour backward compatibilité

### 3. train.py — Refactor Phase 1
- `train_phase1_unimodal()` modifiée pour utiliser deux dataloaders séparés:
  - `face_loader` → entraîne l'encodeur face (EfficientNet-B2) sur CASIA-WebFace
  - `fp_loader` → entraîne l'encodeur fingerprint (Custom CNN) sur SOCOFing
- Alterner entre les deux dans chaque epoch, ou faire deux boucles séparées

### 4. train.py — Refactor Phase 2
- `train_phase2_joint()` utilise `SyntheticPairDataset`
- Les paires synthétiques associent aléatoirement un visage et une empreinte
- Le `UFMLoss` s'applique sur la sortie fusionnée
- Modality masking (30%) continue de s'appliquer

### 5. main.py — CLI
- Ajouter `--face_path` et `--fingerprint_path` aux arguments

### 6. Test end-to-end
- Vérifier que le pipeline s'exécute sans erreur sur un petit sous-ensemble
