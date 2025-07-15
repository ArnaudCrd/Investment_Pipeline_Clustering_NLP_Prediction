"""
SCRIPT PRINCIPAL : Pipeline d’Analyse et Recommandation Boursière
Auteur : Arnaud Chéridi,

Objectif :
Ce script exécute de bout en bout un pipeline complet d’analyse financière et de recommandation d’investissement
(Buy / Hold / Sell) basé sur des données de marché, de nouvelles économiques et d’indicateurs techniques.

Étapes du pipeline :
1. Extraction des données financières et historiques (TP1)
2. Analyse de clustering sur le risque et le rendement (TP2)
3. Classification supervisée Buy/Hold/Sell à horizon 20j (TP3)
4. Régression de la valeur de clôture J+1 avec modèles classiques (TP4)
5. Prédiction J+1 avec Deep Learning (TP5)
6. Scraping de news économiques pertinentes (TP6)
7. Analyse de sentiment via modèle LLM fine-tuné (TP7/TP8)
8. Analyse technique (RSI, MACD, MA/EMA) (TP_complémentaire)
9. Génération automatique d’un rapport PDF synthétique par entreprise

Structure :
- Chargement intelligent des modèles (si absents : entraînement relancé)
- Vérification de la complétude des fichiers
- Gestion des erreurs avec try/except
- Agrégation des signaux dans un format final pour le rapport
- Génération PDF stylisé pour chaque entreprise
"""

import os, joblib, json
import pandas as pd

from datetime import datetime, timedelta

from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import TP1, TP2, TP3, TP4, TP5, TP6, TP7, TP8, TP_complementaire
from generate_pdf import generate_pdf_for_company
from companies import companies


def main():
    key_companies = list(companies.values())

    with open("key_api.env", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

    # Récupération de la clé
    NEWS_API_KEY = os.environ.get("NEWS_API_KEY")

    if not NEWS_API_KEY:
        raise ValueError("Clé API manquante. Assurez-vous que key_api.env contient NEWS_API_KEY.")

    # ────────────────────────────────────────────────────────────────────────
    # 1. Extraction des données (TP1)
    # ────────────────────────────────────────────────────────────────────────
    print("Partie 1 : Extraction des données\n")
    TP1.scrape_ratios()
    TP1.scrape_historical()
    print(f"\nRécupération des données de marchés réalisée\n")

    # ────────────────────────────────────────────────────────────────────────
    # 2. Clustering risque et rendement (TP2)
    # ────────────────────────────────────────────────────────────────────────
    print("Partie 2 : Clustering\n")
    df_ratios = pd.read_csv("historiques_entreprises/ratios_financiers.csv")

    # Clustering sur le risque
    df_risk, X_risk, labels_risk = TP2.cluster_risk_profiles(df_ratios)
    df_risk["Ticker"] = companies
    df_risk["Cluster"] = labels_risk

    # Clustering sur le rendement (corrélation)
    labels_ret, corr_ret, dist_ret = TP2.cluster_return_correlations()
    df_rendement = pd.DataFrame({
        "Ticker": companies,
        "Cluster": labels_ret
    })

    # Construction des dictionnaires de similarité

    # Clustering risque
    clustering_risque = {}
    for cl in df_risk["Cluster"].unique():
        tickers = df_risk[df_risk["Cluster"] == cl]["Ticker"].tolist()
        for t in tickers:
            clustering_risque[t] = [x for x in tickers if x != t]

    # Clustering rendement
    clustering_rendement = {}
    for cl in df_rendement["Cluster"].unique():
        tickers = df_rendement[df_rendement["Cluster"] == cl]["Ticker"].tolist()
        for t in tickers:
            clustering_rendement[t] = [x for x in tickers if x != t]

    print("\nClustering risque et rendement générés avec succès.")

    # ────────────────────────────────────────────────────────────────────────
    # 3. Classification Buy / Hold / Sell (TP3)
    # ────────────────────────────────────────────────────────────────────────
    print("Partie 3 : Classification (achat - vente)\n")
    model_classification_path = "models/best_model_classification.pkl"
    scaler_classification_path = "models/scaler_classification.pkl"

    if not os.path.exists(model_classification_path) or not os.path.exists(scaler_classification_path):
        print("Modèle ou scaler manquant. Entraînement en cours.")
        try:
            TP3.pipeline()
        except Exception as e:
            print("Erreur lors de l'entraînement TP3")
            print("STDOUT :", e.stdout)
            print("STDERR :", e.stderr)
            raise
    else:
        print("Modèle et scaler trouvés.")

    scaler_classification = joblib.load(scaler_classification_path)
    model_classification = joblib.load(model_classification_path)
    print(f"\nModèle de classification extrait\n")

    # ────────────────────────────────────────────────────────────────────────
    # 4. Régression J+1 - Modèles Classiques (TP4)
    # ────────────────────────────────────────────────────────────────────────
    print("Partie 4 : Régression\n")

    # Dictionnaires pour stocker les modèles et scalers
    models_reg = {}
    scalers_reg = {}

    # Vérifie si tous les fichiers existent, sinon lance l'entraînement
    missing_companies = []

    for company in key_companies:
        model_path = f"models/reg/best_model_{company}_reg.pkl"
        scaler_path = f"models/reg/scaler_{company}_reg.pkl"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Modèle/scaler manquant pour {company}")
            missing_companies.append(company)

    if missing_companies:
        print(f"\nModèles manquants détectés pour {len(missing_companies)} entreprise(s). Entraînement en cours...")
        for company in missing_companies:
            try:
                TP4.pipeline_one_company(symbol=company)
            except Exception as e:
                print(f"\nErreur lors de l'entraînement pour {company}")
                print("Exception :", e)
                continue
    else:
        print("Tous les modèles et scalers sont disponibles.")

    # Chargement des modèles et scalers
    for company in companies:
        try:
            model_path = f"models/reg/best_model_{company}_reg.pkl"
            scaler_path = f"models/reg/scaler_{company}_reg.pkl"
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            models_reg[company] = model
            scalers_reg[company] = scaler
        except Exception as e:
            print(f"Erreur lors du chargement pour {company} : {e}")

    print(f"\n{len(models_reg)} modèles chargés avec succès")

    # ────────────────────────────────────────────────────────────────────────
    # 5. Réseaux de Neurones - Deep Learning (TP5)
    # ────────────────────────────────────────────────────────────────────────
    print("Partie 5 : Réseaux de neurones (Deep Learning)\n")
    # Dictionnaires pour stocker les modèles et scalers
    models_dl = {}
    scalers_dl = {}

    # Vérifie si tous les fichiers existent, sinon lance l'entraînement
    missing_models = []

    for company in companies:
        model_path = f"models/dl/best_model_{company}_DL.h5"
        scaler_path = f"models/dl/scaler_{company}_DL.pkl"
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"Modèle ou scaler manquant pour {company}")
            missing_companies.append(company)

    if missing_companies:
        print(f"\nModèles manquants détectés pour {len(missing_companies)} entreprise(s).")
        print("Début de l'entraînement.\n")

        for symbol in missing_companies:
            print(f"Entraînement pour : {symbol}")
            try:
                TP5.compare_models(symbol=symbol)
            except Exception as e:
                print(f"\nErreur lors de l'entraînement du modèle pour {symbol}")
                print("Exception :", e)
                continue
    else:
        print("Tous les modèles et scalers deep learning sont disponibles.")

    # Chargement des modèles et scalers
    for company in companies:
        try:
            model_path = f"models/dl/best_model_{company}_DL.h5"
            scaler_path = f"models/dl/scaler_{company}_DL.pkl"
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            models_dl[company] = model
            scalers_dl[company] = scaler
        except Exception as e:
            print(f"Erreur lors du chargement pour {company} : {e}")

    print(f"\n{len(models_dl)} modèles DL chargés avec succès")

    # ────────────────────────────────────────────────────────────────────────
    # 5 bis. Comparaison TP4 vs TP5 (RMSE) et sélection du meilleur modèle
    # ────────────────────────────────────────────────────────────────────────
    print("\nPartie 5bis : Comparaison TP4 vs TP5 (RMSE) et sélection du meilleur modèle\n")

    rmse_path = "models/rmse_summary.csv"
    best_models = {}

    if not os.path.exists(rmse_path):
        print("Fichier rmse_summary.csv introuvable. Assure-toi que TP4 et TP5 ont bien exécuté leurs sauvegardes.")
    else:
        df = pd.read_csv(rmse_path)

        for company in key_companies:
            subset = df[df["symbol"] == company]
            if subset.empty or subset["rmse"].isnull().any():
                print(f"Données incomplètes pour {company}")
                continue

            best_row = subset.loc[subset["rmse"].idxmin()]
            source = best_row["source"]

            if source == "TP4":
                model = models_reg.get(company)
                scaler = scalers_reg.get(company)
            elif source == "TP5":
                model = models_dl.get(company)
                scaler = scalers_dl.get(company)
            else:
                print(f"Source inconnue pour {company}")
                continue

            if model is None or scaler is None:
                print(f"Modèle ou scaler introuvable pour {company} ({source})")
                continue

            best_models[company] = {
                "source": source,
                "model": model,
                "scaler": scaler,
                "rmse": best_row["rmse"]
            }

    print(f"\n{len(best_models)} modèles finaux sélectionnés")

    # ────────────────────────────────────────────────────────────────────────
    # 6. Scraping des News (TP6)
    # ────────────────────────────────────────────────────────────────────────
    print("Partie 6 : Scrapping des NEWS\n")
    name_companies = list(companies.keys())

    for company in name_companies:
        try:
            TP6.get_news_by_date(company_name = company, api_key = NEWS_API_KEY)  # ou get_news_for_company
        except Exception as e:
            print(f"Erreur pour {company} : {e}")

    print(f"\nRécupération des news réalisée\n")

    # ────────────────────────────────────────────────────────────────────────
    # 7. Analyse de Sentiment - Modèle LLM (TP7/TP8)
    # ────────────────────────────────────────────────────────────────────────
    print("\nPartie 7 : Analyse de sentiment LLM\n")

    MODEL_DIR = "models/LLM"
    MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
    TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer_config.json")
    METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

    # 1. Vérification et entraînement si besoin
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Modèle LLM non trouvé. Lancement de l'entraînement.")
        try:
            TP7.pipeline()
        except Exception as e:
            print("\nUne erreur est survenue lors de l'entraînement du modèle LLM.")
            print("Exception :", e)
            raise
    else:
        print("Modèle LLM fine-tuné détecté.")

    # 2. Chargement du modèle et tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    print(f"\nRécupération du modèle LLM réalisée\n")

    print("\nPartie 8 : Analyse de sentiment LLM\n")

    sentiment_history = {}

    for company in companies:
        try:
            file_path = f"news/{company}_news.json"
            if not os.path.exists(file_path):
                print(f"Fichier introuvable pour {company}")
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            news = []
            for date, articles in data.items():
                for article in articles:
                    article["date"] = date
                    news.append(article)

            today = datetime.today()
            cutoff_date = today - timedelta(days=30)
            last_news = [n for n in news if "date" in n and datetime.strptime(n["date"], "%Y-%m-%d") >= cutoff_date]

            # Si moins de 5 news dans les 30 derniers jours, compléter avec les plus récentes
            if len(last_news) < 5:
                # Trier toutes les news (même au-delà de 30 jours) par date décroissante
                news_sorted = sorted(news, key=lambda x: x["date"], reverse=True)
                # Compléter avec des articles plus anciens
                for n in news_sorted:
                    if n not in last_news:
                        last_news.append(n)
                    if len(last_news) >= 5:
                        break

            texts = [item["title"] for item in last_news if "title" in item]
            dates = [item["date"] for item in last_news if "date" in item]

            if not texts or len(texts) != len(dates):
                print(f"News invalides pour {company}")
                continue

            os.makedirs("sentiments", exist_ok=True)
            preds = TP8.analyze_sentiments_by_company({company: texts}, tokenizer, model)[company]
            sentiment_labels = [ ["Négatif", "Neutre", "Positif"][p] for p in preds ]

            df = pd.DataFrame({
                "date": dates,
                "headline": texts,
                "sentiment_code": preds,
                "sentiment_label": sentiment_labels
            })

            sentiment_history[company] = df

            # sauvegarde CSV
            df.to_csv(f"sentiments/sentiments_{company}.csv", index=False)

        except Exception as e:
            print(f"Erreur pour {company} : {e}")

    print(f"\nRécupération des sentiments réalisée\n")

    # ────────────────────────────────────────────────────────────────────────
    # 8. Analyse Technique (TP_complementaire)
    # ────────────────────────────────────────────────────────────────────────
    print("\nPartie Complémantaire : Analyse technique 📈\n")

    tech_signals = {}

    for company in key_companies:
        try:
            signals = TP_complementaire.technical_analysis(company)
            tech_signals[company] = signals
        except Exception as e:
            print(f"Erreur analyse technique pour {company} : {e}")

    print("\nAnalyse technique effectué\n")

    # ────────────────────────────────────────────────────────────────────────
    # 9. Génération du Rapport PDF final
    # ────────────────────────────────────────────────────────────────────────
    print("\nGénération des rapports PDF\n")
    for ticker in key_companies:
        try:
            generate_pdf_for_company(
                ticker=ticker,
                best_models=best_models,
                model_classification=model_classification,
                scaler_classification=scaler_classification,
                tech_signals=tech_signals,
                clustering_risque=clustering_risque,
                clustering_rendement=clustering_rendement
            )
        except Exception as e:
            print(f"Erreur pour {ticker} : {e}")
    print("\nRapport Générés\n")

if __name__ == "__main__":
    main()