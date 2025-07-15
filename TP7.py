"""
Fine-tuning d’un modèle de sentiment financier

Auteurs : Wayan Crain, Arnaud Chéridi

Description :
Ce script fine-tune des modèles de classification (BERT) pour analyser le
sentiment d’actualités financières. Il utilise deux jeux de données publics,
entraîne plusieurs modèles, et sélectionne automatiquement le meilleur
selon le F1-score pondéré.

Fonctionnalités :
- Chargement et fusion de deux jeux de données de classification financière
- Prétraitement avec tokenisation (transformers)
- Entraînement et évaluation de modèles de type BERT
- Sélection et sauvegarde du meilleur modèle et tokenizer
- Export des métriques de performance
"""

from datasets import load_dataset, concatenate_datasets, DatasetDict
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import os
os.environ["WANDB_DISABLED"] = "true"

# ──────────────────────────────────────────────────────────────────────
# 1. Chargement des jeux de données
# ──────────────────────────────────────────────────────────────────────
def load_and_prepare_datasets():
    """
    Charge et fusionne deux jeux de données d’actualités financières.

    Returns:
        DatasetDict: Dictionnaire contenant deux sous-ensembles :
                     - "train" : ensemble d'entraînement fusionné
                     - "test"  : ensemble de test fusionné
    """
    ds1 = load_dataset("zeroshot/twitter-financial-news-sentiment")
    ds2 = load_dataset("nickmuchi/financial-classification")

    if "label" not in ds2["train"].features:
        ds2 = ds2.rename_column("labels", "label")

    train_data = concatenate_datasets([ds1["train"], ds2["train"]])
    test_data  = concatenate_datasets([ds1["validation"], ds2["test"]])
    return DatasetDict({"train": train_data, "test": test_data})

# ──────────────────────────────────────────────────────────────────────
# 2. Tokenisation
# ──────────────────────────────────────────────────────────────────────
def tokenize_dataset(tokenizer, dataset):
    """
    Applique la tokenisation aux textes des jeux de données.

    Args:
        tokenizer (AutoTokenizer): Tokenizer HuggingFace.
        dataset (DatasetDict): Dictionnaire contenant "train" et "test".

    Returns:
        tuple: (tokenized_train, tokenized_test) datasets formatés pour PyTorch.
    """
    def preprocess(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)
    tokenized_train = dataset["train"].map(preprocess, batched=True)
    tokenized_test  = dataset["test"].map(preprocess, batched=True)

    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_train, tokenized_test

# ──────────────────────────────────────────────────────────────────────
# 3. Entraînement et évaluation
# ──────────────────────────────────────────────────────────────────────
def train_and_evaluate(model_name: str,
                       dataset: DatasetDict,
                       batch_size: int = 16,
                       num_epochs: int = 3):
    """
    Entraîne et évalue un modèle sur le jeu de données fourni.

    Args:
        model_name (str): Nom du modèle pré-entraîné (ex: 'bert-base-uncased').
        dataset (DatasetDict): Jeu de données complet ("train" + "test").
        batch_size (int): Taille de batch pour l'entraînement.
        num_epochs (int): Nombre d’époques.

    Returns:
        tuple: (modèle entraîné, tokenizer, métriques d’évaluation)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    X_train, X_test = tokenize_dataset(tokenizer, dataset)

    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        do_eval=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        prf = precision_recall_fscore_support(labels, preds, average="weighted")
        return {"accuracy": acc, "precision": prf[0], "recall": prf[1], "f1": prf[2]}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=X_train,
        eval_dataset=X_test,
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n📊 Résultats pour {model_name}:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    return model, tokenizer, metrics

# ──────────────────────────────────────────────────────────────────────
# 4. Pipeline principal
# ──────────────────────────────────────────────────────────────────────
def pipeline():
    """
    Exécute tout le pipeline de fine-tuning :
    - Chargement des données
    - Entraînement de plusieurs modèles
    - Sélection du meilleur (via F1-score)
    - Sauvegarde modèle, tokenizer et métriques

    Returns:
        None
    """
    dataset = load_and_prepare_datasets()

    candidates = [
        "bert-base-uncased",
        "yiyanghkust/finbert-tone"
    ]

    best_model = None
    best_tokenizer = None
    best_metrics = None
    best_f1 = -1

    for model_name in candidates:
        model, tokenizer, metrics = train_and_evaluate(model_name, dataset)
        if metrics["eval_f1"] > best_f1:
            best_f1 = metrics["eval_f1"]
            best_model = model
            best_tokenizer = tokenizer
            best_metrics = metrics

    # Sauvegarde du meilleur modèle
    output_dir = "models/LLM"
    os.makedirs(output_dir, exist_ok=True)
    best_model.save_pretrained(output_dir)
    best_tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(best_metrics, f, indent=4)
    print(f"\nMeilleur modèle sauvegardé dans {output_dir} (F1 = {best_f1:.4f})")
