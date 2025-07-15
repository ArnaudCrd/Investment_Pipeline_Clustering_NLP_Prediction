"""
SCRIPT PDF – Génération de rapports boursiers par entreprise
Auteur : Arnaud Chéridi

Objectif :
Générer un rapport PDF professionnel et synthétique pour chaque entreprise, contenant :
  1. Prédiction de la valeur de clôture à J+1 (modèle optimal TP4/TP5)
  2. Conseil classification Buy/Hold/Sell (TP3)
  3. Analyse technique (RSI, MACD, MA/EMA) (TP_complémentaire)
  4. Analyse économique à partir des news et du LLM (TP7/TP8)
  5. Comparaison avec entreprises similaires (clustering TP2)
  6. Conseil final agrégé (modèle + technique + news)

Fonctionnalités :
  - Mise en page A4 stylisée avec titres, images et tags colorés
  - Encodage et affichage sécurisé des textes
  - Intégration automatique des graphiques techniques
  - Agrégation des résultats pour décision finale
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import ta

from fpdf import FPDF
from datetime import timedelta

from TP_complementaire import get_last_features_for_classification
from companies import companies

class InvestmentPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self.safe_text(self.title), ln=True, align='C')
        self.ln(5)

    def safe_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.replace("€", "EUR")
        return text.encode("latin-1", errors="ignore").decode("latin-1")

    def add_image(self, path, x, y, w):
        if os.path.exists(path):
            self.image(path, x=x, y=y, w=w)

    def add_colored_tag(self, label, x, y):
        colors = {"SELL": (255, 0, 0), "HOLD": (255, 165, 0), "BUY": (0, 200, 0)}
        r, g, b = colors.get(label.upper(), (180, 180, 180))
        self.set_xy(x, y)
        self.set_fill_color(r, g, b)
        self.set_font("Helvetica", "B", 12)
        self.cell(40, 10, self.safe_text(label), border=1, align='C', fill=True)

    def add_text_box(self, text, x, y, w):
        self.set_xy(x, y)
        self.set_font("Helvetica", size=10)
        self.multi_cell(w, 5, self.safe_text(text))


def generate_graphs(ticker, df_hist, y_pred, output_dir):
    os.makedirs("pdf_reports/images", exist_ok=True)
    paths = {}

    # Graph 1 : Prévision J+1
    fig, ax = plt.subplots(figsize=(5, 3))
    df_hist['Close'].iloc[-30:].plot(ax=ax, label='Close', color='black')
    x_pred = df_hist.index[-1] + timedelta(days=1)
    ax.scatter([x_pred], [y_pred], color='red', label='Prévision J+1')
    ax.legend()
    ax.set_title("Prix + Prédiction")
    path_pred = os.path.join(output_dir, f"images/{ticker}_prediction.png")
    plt.tight_layout()
    plt.savefig(path_pred)
    plt.close()
    paths['prediction'] = path_pred

    # Graph 2 : RSI / MACD
    fig, ax = plt.subplots(figsize=(5, 3))
    df_hist[['RSI', 'MACD', 'MACD_Signal']].iloc[-30:].plot(ax=ax)
    ax.axhline(70, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(30, color='green', linestyle='--', linewidth=0.5)
    ax.set_title("RSI / MACD")
    path_rsi_macd = os.path.join(output_dir, f"images/{ticker}_rsi_macd.png")
    plt.tight_layout()
    plt.savefig(path_rsi_macd)
    plt.close()
    paths['rsi_macd'] = path_rsi_macd

    # Graph 3 : MA / EMA
    fig, ax = plt.subplots(figsize=(5, 3))
    df_hist['MA50'].iloc[-30:].plot(ax=ax, label='MA50', color='blue')
    df_hist['EMA20'].iloc[-30:].plot(ax=ax, label='EMA20', color='red')
    df_hist['Close'].iloc[-30:].plot(ax=ax, label='Close', linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_title("MA50 / EMA20")
    path_ma_ema = os.path.join(output_dir, f"images/{ticker}_ma_ema.png")
    plt.tight_layout()
    plt.savefig(path_ma_ema)
    plt.close()
    paths['ma_ema'] = path_ma_ema

    return paths


def generate_pdf_for_company(ticker, best_models, model_classification, scaler_classification, tech_signals, clustering_risque, clustering_rendement, output_dir="pdf_reports"):
    os.makedirs(output_dir, exist_ok=True)
    name = companies.get(ticker, ticker)
    pdf = InvestmentPDF(format="A4")
    pdf.title = f"Rapport pour {name} ({ticker})"
    pdf.add_page()

    df = pd.read_csv(f"historiques_entreprises/{ticker}_historique.csv", index_col=0, parse_dates=True).sort_index()
    df['RSI'] = ta.momentum.rsi(df['Close'])
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
    df['MA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)

    model_info = best_models[ticker]
    x = df['Close'].iloc[-30:].values.reshape(-1, 1)
    x_scaled = model_info['scaler'].transform(x).reshape(1, -1)
    y_pred_scaled = model_info['model'].predict(x_scaled)[0]
    y_pred = model_info['scaler'].inverse_transform([[y_pred_scaled]])[0][0]

    img_paths = generate_graphs(ticker, df, y_pred, output_dir)

    x_class = get_last_features_for_classification(ticker)
    conseil_num = model_classification.predict(scaler_classification.transform(x_class))[0]
    conseil_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
    conseil = conseil_map[conseil_num]

    sent_path = f"sentiments/sentiments_{name}.csv"
    if os.path.exists(sent_path):
        df_sent = pd.read_csv(sent_path)
        sentiment = df_sent["sentiment_label"].value_counts().idxmax()
        headlines = df_sent.head(5).apply(lambda row: f"- {row['headline']} ({row['sentiment_label']})", axis=1).tolist()
    else:
        sentiment = "AUCUNE NEWS"
        headlines = ["Aucune news disponible"]

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_xy(10, 30)
    pdf.cell(0, 10, pdf.safe_text("Partie 1 : Analyse par modèle"), ln=True)
    pdf.add_text_box(f"Prévision J+1: {y_pred:.2f}\nConseil modèle: {conseil}", 10, 40, 90)
    pdf.add_image(img_paths['prediction'], x=110, y=40, w=90)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_y(100)
    pdf.cell(0, 10, pdf.safe_text("Partie 2 : Analyse technique"), ln=True)
    signals = "\n".join([f"{ind} : {sig}" for ind, (sig, _) in tech_signals[ticker].items()])
    pdf.add_text_box(signals, 10, 110, 190)
    pdf.add_image(img_paths['rsi_macd'], x=10, y=125, w=90)
    pdf.add_image(img_paths['ma_ema'], x=110, y=125, w=90)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_y(180)
    pdf.cell(0, 10, pdf.safe_text("Partie 3 : Analyse économique"), ln=True)
    news_block = "\n".join(headlines)
    pdf.set_xy(10, 190)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(190, 5, pdf.safe_text(f"Sentiment Global: {sentiment}\n\n{news_block}"))

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_y(230)
    pdf.cell(0, 10, pdf.safe_text("Partie 4 : Entreprises similaires"), ln=True)
    risque = clustering_risque.get(ticker, [])[:7]
    rendement = clustering_rendement.get(ticker, [])[:7]
    risque_names = [companies.get(t, t) for t in risque]
    rendement_names = [companies.get(t, t) for t in rendement]
    cluster_txt = f"Similaires (Risque): {', '.join(risque_names)}\nSimilaires (Rendement): {', '.join(rendement_names)}"
    pdf.add_text_box(cluster_txt, 10, 240, 180)

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_y(255)
    pdf.cell(0, 10, pdf.safe_text("Partie 5 : Conseil final (agrégé)"), ln=True)
    score = conseil_num + {"NEGATIF": 0, "NEUTRE": 1, "POSITIF": 2, "AUCUNE NEWS": 1}.get(sentiment.upper(), 1)
    score += sum([{"SELL": 0, "HOLD": 1, "BUY": 2}.get(v[0].upper(), 1) for v in tech_signals[ticker].values()])
    final_score = score / (2 + len(tech_signals[ticker]))
    final_conseil = "SELL" if final_score < 0.8 else "HOLD" if final_score < 1.6 else "BUY"
    pdf.add_colored_tag(final_conseil, x=10, y=265)

    pdf.output(os.path.join(output_dir, f"rapport_{ticker}.pdf"))
