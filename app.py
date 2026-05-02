import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

st.title("📊 Dashboard Markowitz - BVMT")

st.write("Importer les fichiers Excel BVMT (2021–2025) et construire un portefeuille optimal.")

# Upload fichiers Excel
uploaded_files = st.file_uploader(
    "Importer les fichiers Excel (2021 à 2025)",
    type=["xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_data = []

    for file in uploaded_files:
        df = pd.read_excel(file)
        all_data.append(df)

    data = pd.concat(all_data, ignore_index=True)

    st.subheader("Aperçu des données")
    st.dataframe(data.head())

    # Choix des colonnes
    date_col = st.selectbox("Choisir la colonne Date", data.columns)
    name_col = st.selectbox("Choisir la colonne Banque / Société", data.columns)
    price_col = st.selectbox("Choisir la colonne Cours de clôture", data.columns)

    # Nettoyage
    data = data[[date_col, name_col, price_col]]
    data.columns = ["Date", "Societe", "Close"]

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Close"] = (
        data["Close"]
        .astype(str)
        .str.replace(",", ".")
    )
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")

    data = data.dropna()

    # Tableau des prix
    prices = data.pivot_table(
        index="Date",
        columns="Societe",
        values="Close",
        aggfunc="last"
    )

    prices = prices.sort_index()
    prices = prices.ffill()

    st.subheader("📈 Prix des banques")
    st.dataframe(prices)

    # Choix des banques
    banques = list(prices.columns)

    selected_banques = st.multiselect(
        "Choisir les banques",
        options=banques,
        default=banques[:min(5, len(banques))]
    )

    # Taux sans risque
    rf = st.number_input("Taux sans risque (%)", value=7.5) / 100

    if len(selected_banques) >= 2:
        selected_prices = prices[selected_banques]
        returns = selected_prices.pct_change().dropna()

        mean_returns = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        cov_matrix = returns.cov() * 252

        st.subheader("📊 Indicateurs")
        st.dataframe(pd.DataFrame({
            "Rentabilité": mean_returns,
            "Volatilité": volatility
        }))

        st.subheader("📊 Matrice variance-covariance")
        st.dataframe(cov_matrix)

        # Markowitz
        n = len(selected_banques)

        def port_return(w):
            return np.dot(w, mean_returns)

        def port_vol(w):
            return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

        def neg_sharpe(w):
            return -(port_return(w) - rf) / port_vol(w)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        init = np.ones(n) / n

        result = minimize(neg_sharpe, init, method="SLSQP",
                          bounds=bounds, constraints=constraints)

        weights = result.x

        ret = port_return(weights)
        vol = port_vol(weights)
        sharpe = (ret - rf) / vol

        st.subheader("🚀 Portefeuille optimal (Sharpe max)")

        st.dataframe(pd.DataFrame({
            "Banque": selected_banques,
            "Poids": weights
        }))

        st.write("📈 Rentabilité :", round(ret * 100, 2), "%")
        st.write("⚠️ Risque :", round(vol * 100, 2), "%")
        st.write("⭐ Sharpe :", round(sharpe, 4))

    else:
        st.warning("Choisir au moins 2 banques")

else:
    st.info("Importer les fichiers Excel pour commencer")
