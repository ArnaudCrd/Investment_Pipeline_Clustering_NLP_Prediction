"""
Réseaux de neurones - données boursières J+1
Auteurs : Yanis Aoudjit, Arnaud Chéridi

Description :
Ce script prédit la valeur de clôture (Close) d'une action à J+1
à partir de ses 30 jours précédents (fenêtre glissante).
Il comprend :
- la création de X, y à partir des historiques de prix (importé du TP4)
- l'entraînement modèles de deep learning (MLP, RNN, LSTM)
- l'évaluation via MAE et RMSE
- la sauvegarde automatique du meilleur modèle
"""

import os, warnings, joblib

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error

from TP4 import build_dataset_reg

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ───────────────────────────────────────────────────────────────────────
# 1.  Modèles de deep-learning
# ───────────────────────────────────────────────────────────────────────

def build_mlp_model(input_shape: tuple,
                    hidden_dims: list = [64, 32],
                    dropout_rate: float = 0.1,
                    activation: str = "relu",
                    optimizer: str = "adam",
                    learning_rate: float = 1e-3):
    """
    Construit un modèle MLP (Multi-Layer Perceptron) pour la régression.

    Args:
        input_shape (tuple): Forme de l'entrée (window_size,).
        hidden_dims (list): Nombre de neurones par couche cachée.
        dropout_rate (float): Taux de dropout entre les couches.
        activation (str): Fonction d'activation (ex: "relu").
        optimizer (str): Optimiseur (ex: "adam").
        learning_rate (float): Taux d'apprentissage.

    Returns:
        keras.Model: Modèle compilé prêt à être entraîné.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim, activation=activation))
        if dropout_rate:
            model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}}), loss="mse")
    return model


def build_rnn_model(input_shape: tuple,
                    units: int = 50,
                    dropout_rate: float = 0.1,
                    activation: str = "tanh",
                    optimizer: str = "adam",
                    learning_rate: float = 1e-3):
    """
    Construit un modèle RNN simple pour la régression.

    Args:
        input_shape (tuple): Forme de l'entrée (window_size, 1).
        units (int): Nombre de neurones dans la couche RNN.
        dropout_rate (float): Taux de dropout.
        activation (str): Fonction d'activation (ex: "tanh").
        optimizer (str): Optimiseur (ex: "adam").
        learning_rate (float): Taux d'apprentissage.

    Returns:
        keras.Model: Modèle RNN compilé.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(units, activation=activation, dropout=dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}}), loss="mse")
    return model


def build_lstm_model(input_shape: tuple,
                     units: int = 50,
                     dropout_rate: float = 0.1,
                     activation: str = "tanh",
                     optimizer: str = "adam",
                     learning_rate: float = 1e-3):
    """
    Construit un modèle LSTM pour la régression.

    Args:
        input_shape (tuple): Forme de l'entrée (window_size, 1).
        units (int): Nombre de neurones dans la couche LSTM.
        dropout_rate (float): Taux de dropout.
        activation (str): Fonction d'activation (ex: "tanh").
        optimizer (str): Optimiseur (ex: "adam").
        learning_rate (float): Taux d'apprentissage.

    Returns:
        keras.Model: Modèle LSTM compilé.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.LSTM(units, activation=activation, dropout=dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}}), loss="mse")
    return model


# ────────────────────────────────────────────────────────────────────────
# 2.  Fonction générique : entraînement et reporting
# ────────────────────────────────────────────────────────────────────────

def train_model(model_type: str,
                X_train: np.ndarray,
                y_train: np.ndarray,
                epochs: int = 10,
                batch_size: int = 32,
                **params):
    """
    Entraîne un modèle de type MLP, RNN ou LSTM selon les données fournies.

    Args:
        model_type (str): Type de modèle à entraîner ("MLP", "RNN", "LSTM").
        X_train (np.ndarray): Features d'entraînement.
        y_train (np.ndarray): Cible d'entraînement.
        epochs (int): Nombre d'époques.
        batch_size (int): Taille de batch.
        **params: Hyperparamètres spécifiques au modèle.

    Returns:
        keras.Model: Modèle entraîné.
    """
    if model_type == "MLP":
        model = build_mlp_model(input_shape=(X_train.shape[1],), **params)
        X_tr = X_train
    elif model_type == "RNN":
        model = build_rnn_model(input_shape=(X_train.shape[1], 1), **params)
        X_tr = X_train[..., np.newaxis]
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape=(X_train.shape[1], 1), **params)
        X_tr = X_train[..., np.newaxis]
    else:
        raise ValueError("model_type doit être MLP ou RNN ou LSTM")

    model.fit(X_tr, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model


def predict_model(model: keras.Model,
                  model_type: str,
                  X_test: np.ndarray,
                  y_test: np.ndarray,
                  scaler: BaseEstimator,
                  show_plot: bool = False):
    """
    Prédit les valeurs de test et évalue les performances.

    Args:
        model (keras.Model): Modèle entraîné.
        model_type (str): Type de modèle ("MLP", "RNN", "LSTM").
        X_test (np.ndarray): Données de test.
        y_test (np.ndarray): Cibles réelles.
        scaler (BaseEstimator): Scaler inverse_transform (MinMaxScaler).
        show_plot (bool): Affiche ou non le graphe.

    Returns:
        float: MAE obtenu sur l'ensemble de test.
    """
    X_te = X_test if model_type == "MLP" else X_test[..., np.newaxis]
    y_pred_scaled = model.predict(X_te, verbose=0).ravel()

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n──── {model_type} – Évaluation ────")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("\nPremières prédictions :")
    for i in range(10):
        print(f"{i+1:2d}  préd = {y_pred[i]:.2f}   réel = {y_true[i]:.2f}")

    if show_plot:
        plt.figure(figsize=(10,4))
        plt.plot(y_true, label="Réel", color="red")
        plt.plot(y_pred, label="Préd.", color="blue")
        plt.title(f"{model_type} – Close réel vs prédiction")
        plt.legend()
        plt.show()

    return mae


# ──────────────────────────────────────────────────────────────────────
# 3. Résumé des performances et sauvegarde
# ──────────────────────────────────────────────────────────────────────

def compare_models(symbol: str = "AAPL",
                   window: int = 30,
                   test_size: float = 0.2,
                   epochs: int = 8,
                   batch_size: int = 32):
    """
    Compare différents modèles de deep learning (MLP, RNN, LSTM) pour la prévision
    du prix de clôture à J+1.

    Étapes :
    - Construction du dataset
    - Grille d'hyperparamètres pour chaque modèle
    - Évaluation (MAE) et sélection du meilleur modèle
    - Sauvegarde du modèle final et du scaler
    - Mise à jour du résumé RMSE (fichier CSV)

    Args:
        symbol (str): Ticker boursier de l'entreprise.
        window (int): Taille de la fenêtre temporelle.
        test_size (float): Proportion de l'ensemble de test.
        epochs (int): Nombre d’époques d'entraînement.
        batch_size (int): Taille des batchs d’entraînement.

    Returns:
        None. Sauvegarde les résultats et affiche les métriques.
    """
    path = f"historiques_entreprises/{symbol}_historique.csv"
    X_tr, X_te, y_tr, y_te, scaler = build_dataset_reg(path, window, test_size)

    configs = {
        "MLP":  [
            {"hidden_dims":[64,32],  "dropout_rate":0.1, "learning_rate":1e-3},
            {"hidden_dims":[128,64], "dropout_rate":0.2, "learning_rate":5e-4}
        ],
        "RNN": [
            {"units":50,  "dropout_rate":0.1, "learning_rate":1e-3},
            {"units":100, "dropout_rate":0.2, "learning_rate":5e-4}
        ],
        "LSTM":[
            {"units":50,  "dropout_rate":0.1, "learning_rate":1e-3},
            {"units":100, "dropout_rate":0.2, "learning_rate":5e-4}
        ]
    }

    best = {}
    for mtype, param_list in configs.items():
        best_mae = np.inf
        best_mod, best_par = None, None
        for p in param_list:
            print(f"\n=== {mtype} – entraînement avec {p} ===")
            mdl = train_model(mtype, X_tr, y_tr,
                              epochs=epochs, batch_size=batch_size, **p)
            mae = predict_model(mdl, mtype, X_te, y_te, scaler)
            if mae < best_mae:
                best_mae, best_mod, best_par = mae, mdl, p
        best[mtype] = (best_mod, best_par, best_mae)

    best_model_type = min(best, key=lambda k: best[k][2])
    best_model, best_params, best_mae = best[best_model_type]

    output_dir = os.path.abspath("models/dl")
    os.makedirs(output_dir, exist_ok=True)
    model_path = f"models/best_model_{symbol}_DL.h5"
    scaler_path = f"models/scaler_{symbol}_DL.pkl"
    best_model.save(model_path)
    joblib.dump(scaler, scaler_path)

    X_full = X_tr if best_model_type == "MLP" else X_tr[..., np.newaxis]
    y_pred_scaled = best_model.predict(X_full, verbose=0).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_tr.reshape(-1, 1)).ravel()

    rmse_final = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\nRMSE finale (DL) sur l'ensemble du dataset : {rmse_final:.4f}")

    summary_path = "models/rmse_summary.csv"
    entry = {
        "symbol": symbol,
        "source": "TP5",
        "model": best_model_type,
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

    print(f"\nMeilleur modèle ({best_model_type}, MAE = {best_mae:.4f}) sauvegardé sous : {model_path}")
    print(f"\nScaler sauvegardé sous : {scaler_path}")
