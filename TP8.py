"""
Classification de sentiment – Inférence

Auteurs : Wayan Crain, Arnaud Chéridi

Description :
Ce script applique le modèle fine-tuné de classification de sentiment (TP7)
pour analyser les actualités économiques d’une entreprise.

Fonctionnalités :
- Chargement du modèle entraîné (`models/LLM`)
- Prédiction du sentiment pour chaque texte (positif, neutre, négatif)
- Agrégation des résultats par entreprise

Utilisation :
- Fonction `predict_sentiment` pour un ensemble de textes
- Fonction `analyze_sentiments_by_company` pour un ensemble multi-entreprise
"""


import torch
import numpy as np

# ────────────────────────────────────────────────────────────────────────
# 1. Prédiction du sentiment pour une liste de textes
# ────────────────────────────────────────────────────────────────────────
def predict_sentiment(texts: list, tokenizer: callable, model:callable, return_probs=False):
    """
    Prédit le sentiment associé à une liste de textes (news).

    Args:
        texts (list of str): Liste des textes à analyser.
        tokenizer (AutoTokenizer): Tokenizer associé au modèle.
        model (AutoModelForSequenceClassification): Modèle entraîné.
        return_probs (bool): Si True, retourne aussi les probabilités.

    Returns:
        np.ndarray: Tableau des prédictions (0=négatif, 1=neutre, 2=positif)
        np.ndarray (optionnel): Tableau des probabilités par classe (si return_probs=True)
    """
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    if return_probs:
        return preds, probs
    return preds


# ────────────────────────────────────────────────────────────────────────
# 2. Analyse de sentiment entreprise par entreprise}
# ────────────────────────────────────────────────────────────────────────
def analyze_sentiments_by_company(news_dict: dict, tokenizer: callable, model: callable):
    """
    Applique la prédiction de sentiment à chaque entreprise.

    Args:
        news_dict (dict): Dictionnaire {ticker: [list of news]}
        tokenizer (AutoTokenizer): Tokenizer du modèle fine-tuné.
        model (AutoModelForSequenceClassification): Modèle fine-tuné.

    Returns:
        dict: {ticker: [list of sentiments]} où chaque sentiment ∈ {0, 1, 2}
    """
    results = {}
    for company, news_list in news_dict.items():
        if not news_list:
            results[company] = []
            continue
        sentiments = predict_sentiment(news_list, tokenizer, model)
        results[company] = sentiments.tolist()
    return results
