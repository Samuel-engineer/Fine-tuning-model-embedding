# 🧠 Synthetic BigPatent Similarity

Ce projet explore la **similarité sémantique de triplets textuels** (ancre, positif, négatif) sur un jeu de données inspiré de **BigPatent**. L'objectif est d'entraîner un modèle à distinguer si deux textes techniques sont proches en sens, dans un contexte de classification ou de recherche de brevets.

## 📌 Objectif

Utiliser un modèle `SentenceTransformer` pour apprendre une **fonction d'encodage** de textes techniques. L’entraînement repose sur une **loss de type triplet**, où le modèle apprend à rapprocher sémantiquement un texte "positif" de l’ancre, tout en éloignant un "négatif".

## 🧰 Outils & Technologies

- `Python 3.10+`
- `Hugging Face Datasets`
- `Sentence Transformers`
- `PyTorch`
- `W&B` pour le suivi des expériences

## 🗂 Structure du dataset

Le jeu de données est structuré en triplets :
- `queery`: la requête ou le document de base
- `positive`: un document sémantiquement proche
- `negative`: un document sans lien sémantique direct

Le dataset est découpé en trois splits : `train`, `valid`, `test`.

## 🔍 Entraînement

Le modèle est fine-tuné avec la `TripletLoss`, combiné à un `TripletEvaluator` basé sur la similarité cosinus.  
Un modèle pré-entraîné léger (`GIST-small`) est utilisé comme base.

## ⚙️ Paramètres d'entraînement

```python
{
  "num_train_epochs": 10,
  "per_device_train_batch_size": 16,
  "per_device_eval_batch_size": 10,
  "learning_rate": 1e-5,
}
