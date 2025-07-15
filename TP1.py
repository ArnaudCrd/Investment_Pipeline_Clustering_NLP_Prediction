"""
Extractions de données financières
Auteurs : Yanis Aoudjit, Arnaud Chéridi
Description :
Ce script télécharge deux types de données pour un ensemble d'entreprises cotées :
1. Leurs ratios financiers fondamentaux (ex. PER, ROE, marges) via yfinance.
2. L'historique de leur cours de clôture sur les 5 dernières années.

Les données sont sauvegardées dans des fichiers CSV, afin d'être utilisées pour
le clustering ultérieur des entreprises selon leurs caractéristiques économiques
et boursières.
"""

import os
import yfinance as yf
import pandas as pd

from datetime import datetime
from dateutil.relativedelta import relativedelta

from companies import companies


def scrape_ratios(histo_dir: str = "historiques_entreprises"):
    """
    Récupère les principaux ratios financiers de grandes entreprises via yfinance et les sauvegarde en CSV.

    Args:
       histo_dir (str): Chemin de sauvegarde du fichier CSV.
    """

    ratios = [
        "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
        "trailingEps", "debtToEquity", "currentRatio", "quickRatio",
        "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
    ]
    dict_ratios = {r: [] for r in ratios}

    # 2. Boucle pour récup les ratios pour chaque société
    for nom, symb in companies.items():
        info = yf.Ticker(symb).info or {}

        for r in ratios:
            dict_ratios[r].append(info.get(r))

    df = pd.DataFrame(dict_ratios, index=companies.keys())
    df.index.name = "Société"

    os.makedirs(histo_dir, exist_ok=True)
    chemin = os.path.join(histo_dir, "ratios_financiers.csv")
    df.to_csv(chemin)
    print(f"\nRatios financiers sauvegardés dans {chemin}")


def scrape_historical(histo_dir: str = "historiques_entreprises"):
    """
    Récupère l'historique des cours de bourse (5 ans) de chaque entreprise et calcule le rendement journalier.

    Args:
        histo_dir (str): Dossier de sauvegarde des historiques.
    """

    os.makedirs(histo_dir, exist_ok=True)
    date_fin = datetime.today()
    date_debut = date_fin - relativedelta(years=5)

    for nom, symb in companies.items():
        data = yf.download(
            symb,
            start=date_debut.strftime("%Y-%m-%d"),
            end=date_fin.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False
        )
        if data.empty:
            print(f"Aucune donnée pour {nom} ({symb})")
            continue

        close = data["Close"]
        close_next = close.shift(-1)
        rendement = (close_next - close) / close

        df_hist = pd.concat([close, close_next, rendement], axis=1)
        df_hist.columns = ["Close", "Close_Lendemain", "Rendement"]

        chemin = os.path.join(histo_dir, f"{symb}_historique.csv")
        df_hist.to_csv(chemin)

    print(f"\nHistorique sauvegardé dans le dossier : {chemin}")