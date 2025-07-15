"""
Clustering des entreprises
Auteurs : Yanis Aoudjit, Arnaud Chéridi
Description :
Ce script télécharge deux types de données pour un ensemble d'entreprises cotées :
1. Leurs ratios financiers fondamentaux (ex. PER, ROE, marges) via yfinance.
2. L'historique de leur cours de clôture sur les 5 dernières années.

Description :
Ce script applique trois approches de clustering aux entreprises :
1. K-Means sur les ratios financiers de performance
2. Clustering hiérarchique sur les ratios de risque (endettement, liquidité)
3. Clustering hiérarchique sur les corrélations de rendements boursiers

Chaque méthode est évaluée via le score de silhouette.
Visualisations disponibles : méthode du coude, t-SNE, dendrogrammes.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# ──────────────────────────────────────────────────────────────────────
# 1. Profils financiers – K-Means
# ──────────────────────────────────────────────────────────────────────
def cluster_financial_profiles(df_ratios: pd.DataFrame, k: int = 4, show_plots: bool = False):
    """
    Clustering K-Means des entreprises selon leurs ratios de performance.

    Args:
        df_ratios (pd.DataFrame): Données de ratios financiers.
        k (int): Nombre de clusters.
        show_plots (bool): Affiche les visualisations si True.

    Returns:
        Tuple contenant le DataFrame enrichi, la matrice X standardisée et les labels.
    """
    cols_perf = [
        "forwardPE", "beta", "priceToBook", "priceToSales", "dividendYield",
        "returnOnEquity", "returnOnAssets", "operatingMargins", "profitMargins"
    ]
    df_fin = df_ratios[cols_perf].copy()
    df_fin.dropna(axis=1, how="all", inplace=True)
    df_fin.fillna(df_fin.mean(), inplace=True)
    df_fin = df_fin.loc[:, df_fin.nunique() > 1]

    scaler = StandardScaler()
    X = scaler.fit_transform(df_fin)

    if show_plots:
        inertias = [KMeans(n_clusters=k).fit(X).inertia_ for k in range(1, 9)]
        plt.plot(range(1, 9), inertias, "o-")
        plt.title("Méthode du coude – Profils financiers")
        plt.show()

    model = KMeans(n_clusters=k, random_state=42).fit(X)
    df_fin["cluster_financier"] = model.labels_

    if show_plots:
        vis = TSNE(perplexity=10, random_state=42).fit_transform(X)
        for c in range(k):
            plt.scatter(vis[model.labels_ == c, 0], vis[model.labels_ == c, 1], label=f"Cluster {c}")
        plt.title("t-SNE – Profils financiers")
        plt.legend()
        plt.show()

    return df_fin, X, model.labels_


# ──────────────────────────────────────────────────────────────────────
# 2. Profils de risque – Hiérarchique
# ──────────────────────────────────────────────────────────────────────

def cluster_risk_profiles(df_ratios: pd.DataFrame, n_clusters: int = 3, show_plots: bool = False):
    """
    Clustering hiérarchique des entreprises selon leurs ratios de risque.

    Args:
        df_ratios (pd.DataFrame): Données de ratios financiers.
        n_clusters (int): Nombre de clusters à extraire.
        show_plots (bool): Affiche le dendrogramme si True.

    Returns:
        Tuple contenant le DataFrame enrichi, la matrice X standardisée et les labels.
    """
    cols_risk = ["debtToEquity", "currentRatio", "quickRatio"]
    df_risk = df_ratios[cols_risk].copy()
    df_risk.fillna(df_risk.mean(), inplace=True)
    df_risk = df_risk.loc[:, df_risk.nunique() > 1]

    X = StandardScaler().fit_transform(df_risk)

    model = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    df_risk["cluster_risque"] = model.labels_

    if show_plots:
        linked = linkage(X, method="ward")
        plt.figure(figsize=(10, 4))
        dendrogram(linked, labels=df_risk.index)
        plt.title("Dendrogramme – Profils de risque")
        plt.tight_layout()
        plt.show()

    return df_risk, X, model.labels_


# ──────────────────────────────────────────────────────────────────────
# 3. Corrélations de rendements – Hiérarchique
# ──────────────────────────────────────────────────────────────────────

def cluster_return_correlations(histo_dir: str = "historiques_entreprises", n_clusters: int = 4, show_plots: bool = False):
    """
    Clustering hiérarchique basé sur les corrélations des rendements journaliers.

    Args:
        histo_dir (str): Dossier contenant les CSV d'historiques.
        n_clusters (int): Nombre de clusters à extraire.
        show_plots (bool): Affiche le dendrogramme si True.

    Returns:
        Tuple : labels, matrice de corrélation, matrice des distances
    """
    data = {}
    for path in glob.glob(f"{histo_dir}/*.csv"):
        name = os.path.basename(path).split("_")[0]
        df = pd.read_csv(path, index_col=0)
        if "Rendement" in df.columns:
            data[name] = df["Rendement"]

    df_ret = pd.DataFrame(data).apply(lambda s: s.fillna(s.mean()))
    corr = df_ret.corr()
    dist = 1 - corr

    linked = linkage(squareform(dist.values), method="average")

    if show_plots:
        plt.figure(figsize=(12, 4))
        dendrogram(linked, labels=corr.columns)
        plt.title("Dendrogramme – Corrélations rendements")
        plt.tight_layout()
        plt.show()

    labels = fcluster(linked, t=n_clusters, criterion="maxclust") - 1
    clusters = pd.Series(labels, index=corr.columns, name="cluster_rendements")

    return clusters, corr, dist


# ──────────────────────────────────────────────────────────────────────
# 4. Évaluation et sauvegarde
# ──────────────────────────────────────────────────────────────────────

def evaluate_clustering(X: np.ndarray, labels: np.ndarray, name: str ="") -> float:
    """
    Calcule le score de silhouette pour un clustering donné.

    Args:
        X (np.ndarray): Données standardisées.
        labels (np.ndarray): Labels du clustering.
        name (str): Nom du clustering (pour affichage).

    Returns:
        float: Score de silhouette
    """
    if len(set(labels)) > 1:
        score = silhouette_score(X, labels)
        print(f"Silhouette {name}: {score:.3f}")
        return score
    else:
        print(f"Silhouette {name}: N/A (1 cluster)")
        return np.nan


def evaluate_all_clusterings(X_fin: np.ndarray,
                             labels_fin: np.ndarray,
                             X_risk: np.ndarray,
                             labels_risk: np.ndarray):
    """
    Évalue les performances de clustering via les scores de silhouette.

    Args:
        X_fin (np.ndarray): Données profils financiers.
        labels_fin (np.ndarray): Labels K-Means.
        X_risk (np.ndarray): Données risques.
        labels_risk (np.ndarray): Labels hiérarchique.

    Returns:
        Tuple des scores (KMeans, Hiérarchique, DBSCAN)
    """
    print("Évaluation des scores de silhouette :")

    sil_fin = evaluate_clustering(X_fin, labels_fin, name="KMeans (finance)")
    sil_risk = evaluate_clustering(X_risk, labels_risk, name="Hiérarchique (risque)")

    print(f"Silhouette K-Means (finance)       : {sil_fin:.3f}")
    print(f"Silhouette Hiérarchique (risque)  : {sil_risk:.3f}")

    db = DBSCAN(eps=1.5, min_samples=3).fit(X_fin)
    labels_db = db.labels_
    mask = labels_db != -1
    labels_eff = labels_db[mask]

    if len(np.unique(labels_eff)) > 1:
        sil_db = silhouette_score(X_fin[mask], labels_eff)
        print(f"Silhouette DBSCAN (finance)       : {sil_db:.3f}")
    else:
        sil_db = np.nan
        print("Silhouette DBSCAN (finance)       : N/A (moins de 2 clusters)")

    return sil_fin, sil_risk, sil_db