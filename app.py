import streamlit as st
import pandas as pd
import numpy as np

st.title("Dashboard Markowitz - Banques BVMT")

st.write("Analyse de portefeuille bancaire selon la théorie de Markowitz")

# Données exemple
data = {
    "BIAT": [110, 112, 111, 115, 118],
    "BT": [7, 7.2, 7.1, 7.3, 7.4],
    "AB": [25, 26, 25.5, 26.5, 27],
    "ATB": [3.2, 3.3, 3.25, 3.4, 3.45]
}

prices = pd.DataFrame(data)

st.subheader("1. Prix historiques")
st.dataframe(prices)

returns = prices.pct_change().dropna()

st.subheader("2. Rendements")
st.dataframe(returns)

mean_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252

st.subheader("3. Rentabilité et volatilité annualisées")

metrics = pd.DataFrame({
    "Rentabilité annualisée": mean_returns,
    "Volatilité annualisée": volatility
})

st.dataframe(metrics)

st.subheader("4. Matrice variance-covariance")
st.dataframe(cov_matrix)

st.subheader("5. Choix des banques")

banques = st.multiselect(
    "Choisir les banques",
    options=prices.columns,
    default=list(prices.columns)
)

risk_free_rate = st.number_input(
    "Taux sans risque (%)",
    value=7.5
) / 100

if len(banques) >= 2:
    selected_returns = returns[banques]
    mean_returns = selected_returns.mean() * 252
    cov_matrix = selected_returns.cov() * 252

    n = len(banques)
    weights = np.ones(n) / n

    portfolio_return = np.dot(weights, mean_returns)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    st.subheader("6. Portefeuille équipondéré")

    st.write("Poids de chaque banque :")

    weights_df = pd.DataFrame({
        "Banque": banques,
        "Poids": weights
    })

    st.dataframe(weights_df)

    st.write("Rentabilité du portefeuille :", round(portfolio_return * 100, 2), "%")
    st.write("Risque du portefeuille :", round(portfolio_volatility * 100, 2), "%")
    st.write("Ratio de Sharpe :", round(sharpe_ratio, 4))

else:
    st.warning("Choisis au moins deux banques.")
