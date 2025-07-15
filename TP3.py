"""
Classification boursière
Auteurs : Yanis Aoudjit, Arnaud Chéridi

Description :
Ce script applique plusieurs algorithmes de classification pour prédire une consigne Buy / Hold / Sell à horizon 20 jours sur des actions.
Le pipeline inclut :
- la génération des labels selon un rendement à 20 jours
- l'ajout d'indicateurs techniques (TA)
- la construction du dataset
- l'entraînement et évaluation de plusieurs modèles :
  Random Forest, XGBoost, KNN, SVM linéaire, Régression logistique
- la sauvegarde automatique du meilleur modèle en .pkl
"""

import glob, os, warnings, joblib
import pandas as pd
import numpy as np
import ta

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from grid import grid_rf, grid_knn, grid_svm, grid_xgb, grid_logreg

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# 1. Création du DataFrame et des features
# ──────────────────────────────────────────────────────────────────────

def create_labels_one_company(csv_path: str) -> pd.DataFrame:
    """
    Crée les labels Buy (2), Hold (1), Sell (0) sur 20 jours pour une entreprise.

    Args: csv_path (str): chemin du fichier CSV de l'entreprise

    Returns: df (pd.DataFrame): dataframe avec les labels et les features (Close_Horizon, Horizon_Return, Symbol)
    """
    symb = os.path.basename(csv_path).split("_")[0]
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    df["Close_Horizon"] = df["Close"].shift(-20)
    df["Horizon_Return"] = (df["Close_Horizon"] - df["Close"]) / df["Close"]
    df["Label"] = np.where(df["Horizon_Return"] > 0.05, 2,
                    np.where(df["Horizon_Return"] < -0.05, 0, 1))
    df["Symbol"] = symb
    return df


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute des indicateurs techniques classiques (SMA, EMA, RSI, MACD, Bollinger...).

    Args: df (pd.DataFrame): dataframe avec les features (Close_Horizon, Horizon_Return, Symbol)

    Returns: df (pd.DataFrame): dataframe avec les features et les indicateurs techniques
    """
    close = df["Close"]
    df["SMA20"] = ta.trend.sma_indicator(close, window=20)
    df["EMA20"] = ta.trend.ema_indicator(close, window=20)
    df["RSI14"] = ta.momentum.rsi(close, window=14)
    df["MACD"] = ta.trend.macd(close)
    df["MACD_Signal"] = ta.trend.macd_signal(close)
    boll = ta.volatility.BollingerBands(close, window=20)
    df["Bollinger_High"] = boll.bollinger_hband()
    df["Bollinger_Low"] = boll.bollinger_lband()
    df["Rolling_Vol20"] = close.pct_change().rolling(window=20).std()
    df["ROC10"] = ta.momentum.roc(close, window=10)
    return df


def build_dataset(folder: str = "historiques_entreprises"):
    """
    Construit le dataset global multi-entreprises à partir des historiques CSV.

    Args: folder (str): chemin du dossier contenant les CSV des historiques

    Returns:
        X (pd.DataFrame): features
        y (pd.Series): labels Buy/Hold/Sell
    """
    frames = []
    for path in glob.glob(os.path.join(folder, "*.csv")):
        if "ratios_financiers.csv" in path:
            continue
        df = create_labels_one_company(path)
        df = add_ta_features(df)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True).dropna()

    y = full["Label"].astype(int)
    drop_cols = ["Label", "Close_Horizon", "Horizon_Return", "Symbol"]
    if "Next Day Close" in full.columns:
        drop_cols.append("Next Day Close")
    X = full.drop(columns=drop_cols, errors="ignore")
    return X, y


def train_test_scaled(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """
   Splitte le jeu de données en train/test et applique une standardisation.

   Args:
       X (pd.DataFrame): Features.
       y (pd.Series): Labels.
       test_size (float): Proportion de test.
       random_state (int): Graine aléatoire pour reproductibilité.

   Returns:
       Tuple:
           - X_tr (np.ndarray): Données d'entraînement standardisées.
           - X_te (np.ndarray): Données de test standardisées.
           - y_tr (pd.Series): Labels d'entraînement.
           - y_te (pd.Series): Labels de test.
           - scaler (StandardScaler): Scaler ajusté.
   """
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te, scaler


# ──────────────────────────────────────────────────────────────────────
# 2. Modèles
# ──────────────────────────────────────────────────────────────────────


def rf_classifier(X_tr: np.ndarray, y_tr: pd.Series):
    """
    Entraîne un Random Forest avec recherche de grille.

    Args:
        X_tr (np.ndarray): Données d'entraînement standardisées.
        y_tr (pd.Series): Labels d'entraînement.

    Returns:
        Tuple:
            - best_estimator_ (RandomForestClassifier): Modèle entraîné.
            - best_params_ (dict): Meilleurs hyperparamètres trouvés.
            - best_score_ (float): Score moyen de validation croisée.
    """
    gs = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42),
                      grid_rf, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def xgb_classifier(X_tr: np.ndarray, y_tr: pd.Series):
    """
    Entraîne un modèle XGBoost avec recherche de grille hyperparamètres.

    Args:
        X_tr (np.ndarray): Données d'entraînement standardisées.
        y_tr (pd.Series): Labels d'entraînement.

    Returns:
        Tuple:
            - best_estimator_ (XGBClassifier): Modèle entraîné avec les meilleurs hyperparamètres.
            - best_params_ (dict): Dictionnaire des meilleurs paramètres.
            - best_score_ (float): Score de validation croisée (accuracy).
    """
    gs = GridSearchCV(
        XGBClassifier(num_class=3, subsample=0.8, colsample_bytree=0.8, random_state=42),
        grid_xgb, cv=3, scoring="accuracy", n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def knn_classifier(X_tr: np.ndarray, y_tr: pd.Series):
    """
    Entraîne un classifieur K plus proches voisins avec recherche de grille.

    Args:
        X_tr (np.ndarray): Données d'entraînement standardisées.
        y_tr (pd.Series): Labels d'entraînement.

    Returns:
        Tuple:
            - best_estimator_ (KNeighborsClassifier): Meilleur modèle KNN.
            - best_params_ (dict): Paramètres optimaux choisis par GridSearch.
            - best_score_ (float): Score de validation croisée.
    """
    gs = GridSearchCV(KNeighborsClassifier(), grid_knn, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def svm_classifier(X_tr: np.ndarray, y_tr: pd.Series):
    """
    Entraîne un SVM linéaire avec pondération des classes.

    Args:
        X_tr (np.ndarray): Données d'entraînement standardisées.
        y_tr (pd.Series): Labels d'entraînement.

    Returns:
        Tuple:
            - best_estimator_ (SVC): SVM entraîné avec les meilleurs paramètres.
            - best_params_ (dict): Hyperparamètres sélectionnés.
            - best_score_ (float): Score de validation croisée.
    """
    gs = GridSearchCV(SVC(kernel="linear", max_iter=10000, class_weight="balanced"),
                      grid_svm, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def logreg_classifier(X_tr: np.ndarray, y_tr: pd.Series):
    """
    Entraîne une régression logistique avec pondération des classes et recherche d'hyperparamètres.

    Args:
        X_tr (np.ndarray): Données d'entraînement standardisées.
        y_tr (pd.Series): Labels d'entraînement.

    Returns:
        Tuple:
            - best_estimator_ (LogisticRegression): Modèle entraîné.
            - best_params_ (dict): Meilleurs hyperparamètres choisis.
            - best_score_ (float): Moyenne des scores de validation croisée.
    """
    gs = GridSearchCV(LogisticRegression(max_iter=1000, class_weight="balanced"),
                      grid_logreg, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


# ────────────────────────────────────────────────────────────────────────
# 3.  Fonction générique : entraînement et reporting
# ────────────────────────────────────────────────────────────────────────

def run_and_report(name: str,
                   func,
                   X_tr: np.ndarray,
                   X_te: np.ndarray,
                   y_tr: pd.Series,
                   y_te: pd.Series) -> dict:
    """
    Entraîne un modèle, affiche le rapport de classification et retourne les performances.

    Args:
        name (str): Nom du modèle.
        func (Callable): Fonction d'entraînement renvoyant (modèle, paramètres, score CV).
        X_tr (np.ndarray): Données d'entraînement.
        X_te (np.ndarray): Données de test.
        y_tr (pd.Series): Labels d'entraînement.
        y_te (pd.Series): Labels de test.

    Returns:
        dict: Résultats (modèle, accuracy test, accuracy CV, nom).
    """
    model, params, cv_score = func(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)

    print(f"\n──── {name} ────")
    if params:
        print("Meilleurs paramètres :", params)
    print(classification_report(y_te, y_pred, digits=3))
    print(f"Accur. test         : {acc:.3f}")
    if cv_score is not None:
        print(f"Accur. CV (approx.) : {cv_score:.3f}")

    return {"Modèle": name, "Acc_CV": cv_score, "Acc_test": acc, "model": model}


# ──────────────────────────────────────────────────────────────────────
# 4. Résumé des performances et sauvegarde
# ──────────────────────────────────────────────────────────────────────

def pipeline(folder: str = "historiques_entreprises", output_dir: str = "models") -> None:
    """
    Exécute l'ensemble du pipeline de classification :
    - construction du dataset multi-entreprises
    - split et standardisation
    - entraînement et évaluation de 5 modèles
    - sélection du meilleur selon la CV
    - sauvegarde du meilleur modèle et du scaler

    Args:
        folder (str): Dossier contenant les fichiers CSV d'historique.
        output_dir (str): Dossier de sauvegarde des modèles.
    """
    os.makedirs(output_dir, exist_ok=True)
    X, y = build_dataset(folder)
    X_tr, X_te, y_tr, y_te, scaler = train_test_scaled(X, y)

    algos = [
        ("Random Forest", rf_classifier),
        ("XGBoost", xgb_classifier),
        ("KNN", knn_classifier),
        ("SVM", svm_classifier),
        ("Régression log", logreg_classifier)
    ]

    results = []
    best_model_func = None
    best_cv_score = -np.inf

    for name, func in algos:
        result = run_and_report(name, func, X_tr, X_te, y_tr, y_te)
        results.append(result)

        if result["Acc_CV"] is not None and result["Acc_CV"] > best_cv_score:
            best_cv_score = result["Acc_CV"]
            best_model_func = func

    df_res = pd.DataFrame(results).set_index("Modèle")
    print("\n==== Résumé des performances ====")
    print(df_res.to_string())


    if best_model_func is not None:
        full_scaler = StandardScaler().fit(X)
        full_scaled = full_scaler.transform(X)
        retrained_model, _, _ = best_model_func(full_scaled, y)
        joblib.dump(retrained_model, os.path.join(output_dir, "best_model_classification.pkl"))
        joblib.dump(full_scaler, os.path.join(output_dir, "scaler_classification.pkl"))
        print(f"\nMeilleur modèle réentraîné sur l'ensemble du dataset → sauvegardé sous best_model_classification.pkl")
        print(f"\nScaler complet sauvegardé sous : scaler_classification.pkl")
    else:
        raise ValueError("\nLe meilleur modèle n'a pas pu être défini")

