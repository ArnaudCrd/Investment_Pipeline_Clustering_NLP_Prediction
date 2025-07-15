"""
Récupération des actualités financières d'une entreprise via NewsAPI

Auteurs : Wayan Crain, Arnaud Chéridi

Description :
Ce script interroge l'API NewsAPI pour récupérer les 10 derniers jours d'actualités
concernant une entreprise donnée (au sens du mot-clé dans titre ou description),
filtre les doublons, et sauvegarde les résultats dans un fichier JSON
au format {date: [articles]}.

Fonctionnalités :
- Chargement des news existantes depuis un fichier JSON (évite les doublons)
- Requête API avec filtre de sources financières
- Ajout des nouveaux articles pertinents
- Sauvegarde du fichier {entreprise}_news.json
"""

import requests
import json
from datetime import datetime, timedelta
import os

def load_existing_news(company_name: str):
    """
   Charge les articles existants pour une entreprise depuis un fichier JSON local.

   Args:
       company_name (str): Nom de l'entreprise (ex: "Apple").

   Returns:
       dict: Dictionnaire des articles sous la forme {date (str): [articles (dict)]}.
             Si aucun fichier n'existe, retourne un dictionnaire vide.
   """
    filename = f"news/{company_name}_news.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_news_by_date(company_name: str, api_key: str):
    """
   Récupère les articles récents pour une entreprise via l'API NewsAPI,
   filtre les articles pertinents (titre ou description contient le nom de l'entreprise),
   et met à jour le fichier JSON local sans doublons.

   Args:
       company_name (str): Nom de l'entreprise ciblée.
       api_key (str): Clé d'API valide pour accéder à https://newsapi.org.

   Returns:
       None: Les articles sont sauvegardés localement dans "news/{company_name}_news.json".
             Un message affiche le nombre total d'articles enregistrés.
   """
    url = 'https://newsapi.org/v2/everything'
    last_day = datetime.today().strftime('%Y-%m-%d')
    first_day = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')

    params = {
        "q": company_name,
        "from": first_day,
        "to": last_day,
        "language": "en",
        "pageSize": 100,
        "sources": 'financial-post, the-wall-street-journal, bloomberg, the-washington-post, australian-financial-review, bbc-news, cnn',
        "apiKey": api_key
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Erreur API ({response.status_code}) : {response.text}")
        return

    articles = response.json().get("articles", [])
    print(f"{len(articles)} articles récupérés pour {company_name}.")

    existing_news = load_existing_news(company_name)
    new_news = existing_news.copy()

    for article in articles:
        title = article.get("title", "")
        description = article.get("description", "")
        date = article.get("publishedAt", "").split("T")[0]
        source = article.get("source", {}).get("name", "")

        # Vérification présence du nom dans le titre ou la description
        if company_name.lower() not in (title + description).lower():
            continue

        # Éviter les doublons sur les titres déjà présents
        daily_news = new_news.get(date, [])
        if any(title == a["title"] for a in daily_news):
            continue

        # Ajout
        news_item = {
            "title": title,
            "description": description,
            "source": source,
            "date": date
        }
        new_news.setdefault(date, []).append(news_item)

    # Sauvegarde du fichier JSON mis à jour
    filename = f"{company_name}_news.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(new_news, f, indent=2, ensure_ascii=False)

    print(f"{company_name} : {sum(len(v) for v in new_news.values())} articles enregistrés.")
