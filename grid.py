# TP2 - Classification
grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20],
    "max_features": ["sqrt"]
}

grid_xgb = {
    "n_estimators": [300],
    "max_depth": [6],
    "learning_rate": [0.1]
}

grid_knn = {
    "n_neighbors": [5, 10, 15, 20],
    "weights": ["uniform", "distance"]
}

grid_svm = {"C": [0.1, 1, 10]}

grid_logreg = {"C": [0.1, 1, 10]}

# TP3 - Régression
grid_xgb_reg = {
        "n_estimators": [300, 500],
        "max_depth":    [3, 6],
        "learning_rate": [0.05, 0.1]
    }

grid_rf_reg = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20]
}

grid_knn_reg = {
    "n_neighbors": [5, 10, 15, 20],
    "weights": ["uniform", "distance"]
}