"""
Analyse technique et classification boursière
Auteur : Arnaud Chéridi

Description :
Ce module implémente une série d'outils pour l'analyse technique des actions
et la génération de signaux Buy/Hold/Sell, à partir d'indicateurs comme le RSI,
le MACD et les moyennes mobiles. Il contient également une fonction pour
extraire les dernières features nécessaires à la classification supervisée.
"""


import pandas as pd
from TP3 import create_labels_one_company, add_ta_features
import os

def compute_indicators(df: pd.DataFrame,
                       short_window: int = 12,
                       long_window: int = 26,
                       signal_window: int = 9) -> pd.DataFrame:
    """
    Calcule les indicateurs techniques RSI, MACD, MA50, EMA20.

    Args:
      df (pd.DataFrame): DataFrame contenant une colonne 'Close'
      short_window (int): Période pour EMA courte (MACD)
      long_window (int): Période pour EMA longue (MACD)
      signal_window (int): Période de l'EMA de signal (MACD)

    Returns:
      pd.DataFrame: DataFrame avec les indicateurs ajoutés
    """
    close = df['Close']

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=short_window, adjust=False).mean()
    ema26 = close.ewm(span=long_window, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    ma50 = close.rolling(50).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()

    df["RSI"] = rsi
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MA50"] = ma50
    df["EMA20"] = ema20
    return df

def generate_signals(df: pd.DataFrame) -> dict:
    """
    Génère les signaux buy/hold/sell basés sur les indicateurs RSI, MACD, MA/EMA.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes RSI, MACD, MACD_Signal, MA50, EMA20

    Returns:
        dict: Dictionnaire {nom_indicateur: (signal, commentaire)}
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    signals = {}

    # RSI
    rsi = latest["RSI"]
    if rsi > 70:
        signals["RSI"] = ("sell", "RSI > 70 : sur-achat, possible retournement baissier")
    elif rsi < 30:
        signals["RSI"] = ("buy", "RSI < 30 : sur-vente, possible retournement haussier")
    else:
        signals["RSI"] = ("hold", "RSI neutre")

    # MACD
    if latest["MACD"] > latest["MACD_Signal"] and prev["MACD"] <= prev["MACD_Signal"]:
        signals["MACD"] = ("buy", "Croisement MACD haussier")
    elif latest["MACD"] < latest["MACD_Signal"] and prev["MACD"] >= prev["MACD_Signal"]:
        signals["MACD"] = ("sell", "Croisement MACD baissier")
    else:
        signals["MACD"] = ("hold", "Pas de croisement MACD")

    # MA/EMA
    if latest["EMA20"] > latest["MA50"] and prev["EMA20"] <= prev["MA50"]:
        signals["MA/EMA"] = ("buy", "Croisement EMA20 au-dessus de MA50")
    elif latest["EMA20"] < latest["MA50"] and prev["EMA20"] >= prev["MA50"]:
        signals["MA/EMA"] = ("sell", "Croisement EMA20 en-dessous de MA50")
    else:
        signals["MA/EMA"] = ("hold", "Pas de croisement")

    return signals

def technical_analysis(symbol: str, folder: str = "historiques_entreprises") -> dict:
    """
    Calcule les indicateurs techniques et génère les signaux d’analyse pour une entreprise.

    Args:
        symbol (str): Ticker de l’entreprise (ex. "AAPL")
        folder (str): Dossier contenant les historiques CSV

    Returns:
        dict: Signaux générés par les indicateurs {nom_indicateur: (signal, commentaire)}
    """
    csv_path = os.path.join(folder, f"{symbol}_historique.csv")
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    df = compute_indicators(df)
    signals = generate_signals(df)
    return signals

def get_last_features_for_classification(ticker: str):
    """
    Extrait les dernières features disponibles pour prédire le label Buy/Hold/Sell.

    Args:
        ticker (str): Ticker de l’entreprise (ex. "AAPL")

    Returns:
        pd.DataFrame: DataFrame contenant les features de la dernière date (1 ligne)
    """
    path = f"historiques_entreprises/{ticker}_historique.csv"
    df = create_labels_one_company(path)
    df = add_ta_features(df).dropna()

    drop_cols = ["Label", "Close_Horizon", "Horizon_Return", "Symbol"]
    if "Next Day Close" in df.columns:
        drop_cols.append("Next Day Close")
    X = df.drop(columns=drop_cols, errors="ignore")

    return X.iloc[[-1]]  # dernière ligne pour la prédiction