"""
Régression - données boursières J+1
Auteurs : Yanis Aoudjit, Arnaud Chéridi

Description :
Ce script prédit la valeur de clôture (Close) d'une action à J+1
à partir de ses 30 jours précédents (fenêtre glissante).
Il comprend :
- la création de X, y à partir des historiques de prix
- l'entraînement de plusieurs modèles de régressions (XGB, RF, KNN, LR)
- l'évaluation via MAE et RMSE
- la sauvegarde automatique du meilleur modèle
"""

import os, warnings, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

from grid import grid_knn_reg, grid_rf_reg, grid_xgb_reg

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────
# 1.  Fonctions utilitaires – création du DataFrame
# ────────────────────────────────────────────────────────────────────────

def create_target_features(arr: np.ndarray = np.nan, n: int = 30):
    """
    Génère les séquences d'entraînement X et les valeurs cibles y
    à partir d'une série 1D (ex : prix de clôture).

    Args:
        arr (np.ndarray): Série de prix sous forme (N, 1).
        n (int): Taille de la fenêtre d'observation.

    Returns:
        Tuple (X, y):
            X : np.ndarray (N-n, n)
            y : np.ndarray (N-n,)
    """
    x, y = [], []
    for i in range(n, len(arr)):
        x.append(arr[i-n:i, 0])
        y.append(arr[i, 0])
    return np.array(x), np.array(y)


def build_dataset_reg(csv_path: str ="", window: int = 30, test_size: float = 0.2):
    """
    Construit les jeux d'entraînement et de test pour la régression.

    Args:
        csv_path (str): Chemin vers le fichier CSV historique.
        window (int): Taille de la fenêtre (nombre de jours précédents).
        test_size (float): Proportion du dataset réservé au test.

    Returns:
        Tuple: X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(csv_path, index_col=0).sort_index()
    close = df["Close"].values.reshape(-1, 1)

    split_idx = int(len(close) * (1 - test_size))
    train_close, test_close = close[:split_idx], close[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_close)
    test_scaled = scaler.transform(test_close)

    X_train, y_train = create_target_features(train_scaled, window)
    X_test,  y_test  = create_target_features(test_scaled,  window)

    return X_train, X_test, y_train.ravel(), y_test.ravel(), scaler


# ────────────────────────────────────────────────────────────────────────
# 2.  Modèles de régression
# ────────────────────────────────────────────────────────────────────────

def xgb_reg(X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Entraîne un modèle XGBoost Regressor avec recherche d'hyperparamètres (grille réduite).

    Args:
        X_tr (np.ndarray): Données d'entraînement (features).
        y_tr (np.ndarray): Données d'entraînement (target).

    Returns:
        Tuple: meilleur modèle entraîné, dictionnaire des meilleurs hyperparamètres
    """
    gs = GridSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=42),
        grid_xgb_reg, cv=2, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def rf_reg(X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Entraîne un modèle RandomForestRegressor avec recherche d'hyperparamètres (grille réduite).

    Args:
        X_tr (np.ndarray): Données d'entraînement (features).
        y_tr (np.ndarray): Données d'entraînement (target).

    Returns:
        Tuple: meilleur modèle entraîné, dictionnaire des meilleurs hyperparamètres
    """
    gs = GridSearchCV(
        RandomForestRegressor(random_state=42),
        grid_rf_reg, cv=2, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def knn_reg(X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Entraîne un modèle KNeighborsRegressor avec recherche d'hyperparamètres (grille réduite).

    Args:
        X_tr (np.ndarray): Données d'entraînement (features).
        y_tr (np.ndarray): Données d'entraînement (target).

    Returns:
        Tuple: (meilleur modèle entraîné, dictionnaire des meilleurs hyperparamètres)
    """
    gs = GridSearchCV(
        KNeighborsRegressor(),
        grid_knn_reg, cv=2, n_jobs=-1, verbose=0
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_, gs.best_params_


def lin_reg(X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Entraîne un modèle de régression linéaire sans recherche d'hyperparamètres.

    Args:
        X_tr (np.ndarray): Données d'entraînement (features).
        y_tr (np.ndarray): Données d'entraînement (target).

    Returns:
        Tuple: (modèle entraîné, dictionnaire vide)
    """
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    return model, {}


# ────────────────────────────────────────────────────────────────────────
# 3.  Fonction générique : entraînement et reporting
# ────────────────────────────────────────────────────────────────────────

def train_and_evaluate(name: str,
                       trainer: callable,
                       X_tr: np.ndarray,
                       X_te: np.ndarray,
                       y_tr: np.ndarray,
                       y_te: np.ndarray,
                       scaler: BaseEstimator,
                       close_series: np.ndarray,
                       window: int,
                       show_plot: bool = False):
    """
        Entraîne un modèle de régression, évalue ses performances et affiche les métriques principales.

        Args:
            name (str): Nom du modèle pour l'affichage.
            trainer (callable): Fonction de training retournant un modèle et ses paramètres.
            X_tr (np.ndarray): Données d'entraînement (features).
            X_te (np.ndarray): Données de test (features).
            y_tr (np.ndarray): Cibles d'entraînement.
            y_te (np.ndarray): Cibles de test.
            scaler (BaseEstimator): Scaler utilisé pour l'inversion.
            close_series (np.ndarray): Série complète des valeurs réelles pour le tracé.
            window (int): Fenêtre temporelle utilisée pour construire les features.
            show_plot (bool): Affiche le graphe de prédiction si True.

        Returns:
            dict: Dictionnaire contenant le nom du modèle, MAE et RMSE.
        """
    model, params = trainer(X_tr, y_tr)

    y_pred_scaled = model.predict(X_te)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(y_te.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n──── {name} ────")
    if params: print("Meilleurs paramètres :", params)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")

    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(close_series)), close_series, color="red", label="Valeurs réelles")
        ax.plot(
            range(len(y_tr)+window, len(y_tr)+window+len(y_pred)),
            y_pred, color="blue", label=f"{name} préd."
        )
        ax.set_title(f"{name} – Close réel vs prédiction")
        ax.legend()
        plt.show()

    return {"Modèle": name, "MAE": mae, "RMSE": rmse}


# ──────────────────────────────────────────────────────────────────────
# 4. Résumé des performances et sauvegarde
# ──────────────────────────────────────────────────────────────────────

def pipeline_one_company(symbol: str = "AAPL",
                                    window: int = 30,
                                    folder: str = "historiques_entreprises",
                                    output_dir: str = "models/reg"):
    """
    Pipeline complet de régression pour une entreprise donnée.

    Étapes :
    1. Chargement des données de clôture d'une entreprise.
    2. Création des features avec fenêtre temporelle.
    3. Entraînement et évaluation de plusieurs modèles (XGBoost, RF, KNN, LR).
    4. Sélection du meilleur modèle selon la RMSE.
    5. Réentraînement du modèle sur l'ensemble des données.
    6. Sauvegarde du modèle, scaler et mise à jour d’un fichier résumé CSV.

    Args:
        symbol (str): Ticker de l’entreprise (ex: "AAPL").
        window (int): Taille de la fenêtre temporelle utilisée pour les features.
        folder (str): Dossier contenant les fichiers CSV.
        output_dir (str): Dossier où sauvegarder le modèle et le scaler.

    Returns:
        None
    """

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(folder, f"{symbol}_historique.csv")
    X_tr, X_te, y_tr, y_te, scaler = build_dataset_reg(csv_path, window)

    close_full = pd.read_csv(csv_path, index_col=0)["Close"].values

    algos = [
        ("XGBoost",       xgb_reg),
        ("Random Forest", rf_reg),
        ("KNN",           knn_reg),
        ("Régression linéaire", lin_reg)
    ]

    results = []
    models = []

    for name, trainer in algos:
        model_result = train_and_evaluate(name, trainer, X_tr, X_te, y_tr, y_te,
                                          scaler, close_full, window)
        model, _ = trainer(X_tr, y_tr)
        results.append(model_result)
        models.append((name, model))

    df_res = pd.DataFrame(results).set_index("Modèle")
    print("\n==== Récapitulatif –", symbol, "====")
    print(df_res.to_string())

    best_model_name = df_res["RMSE"].idxmin()
    best_trainer = dict(algos)[best_model_name]

    df = pd.read_csv(csv_path, index_col=0).sort_index()
    close = df["Close"].shift(-1).dropna().values.reshape(-1, 1)
    scaler_full = MinMaxScaler(feature_range=(0, 1))
    scaled_full = scaler_full.fit_transform(close)

    X_full, y_full = create_target_features(scaled_full, window)
    X_full, y_full = X_full, y_full.ravel()

    final_model, _ = best_trainer(X_full, y_full)

    y_pred_scaled = final_model.predict(X_full).reshape(-1, 1)
    y_pred = scaler_full.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_full.inverse_transform(y_full.reshape(-1, 1)).ravel()

    rmse_final = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nRMSE finale sur l'ensemble du dataset : {rmse_final:.4f}")

    summary_path = "models/rmse_summary.csv"
    entry = {
        "symbol": symbol,
        "source": "TP4",
        "model": best_model_name,
        "rmse": rmse_final
    }

    if os.path.exists(summary_path):
        df_summary = pd.read_csv(summary_path)
        df_summary = df_summary[~((df_summary["symbol"] == symbol) & (df_summary["source"] == entry["source"]))]
        df_summary = pd.concat([df_summary, pd.DataFrame([entry])], ignore_index=True)
    else:
        df_summary = pd.DataFrame([entry])

    df_summary.to_csv(summary_path, index=False)
    print(f"Résumé mis à jour dans {summary_path}")

    model_path = os.path.join(output_dir, f"best_model_{symbol}_reg.pkl")
    scaler_path = os.path.join(output_dir, f"scaler_{symbol}_reg.pkl")
    joblib.dump(final_model, model_path)
    joblib.dump(scaler_full, scaler_path)

    print(f"\nMeilleur modèle ({best_model_name}) réentraîné sur tout le dataset → sauvegardé sous {model_path}")
    print(f"\nScaler complet sauvegardé sous : {scaler_path}")
