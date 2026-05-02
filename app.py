import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

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

    def portfolio_return(weights):
        return np.dot(weights, mean_returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def negative_sharpe(weights):
        ret = portfolio_return(weights)
        vol = portfolio_volatility(weights)
        return -(ret - risk_free_rate) / vol

    constraints = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }

    bounds = tuple((0, 1) for i in range(n))
    initial_weights = np.ones(n) / n

    result = minimize(
        negative_sharpe,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x

    optimal_return = portfolio_return(optimal_weights)
    optimal_volatility = portfolio_volatility(optimal_weights)
    optimal_sharpe = (optimal_return - risk_free_rate) / optimal_volatility

    st.subheader("6. Portefeuille optimal selon le ratio de Sharpe")

    weights_df = pd.DataFrame({
        "Banque": banques,
        "Poids optimal": optimal_weights
    })

    st.dataframe(weights_df)

    st.write("Rentabilité optimale :", round(optimal_return * 100, 2), "%")
    st.write("Risque optimal :", round(optimal_volatility * 100, 2), "%")
    st.write("Ratio de Sharpe maximal :", round(optimal_sharpe, 4))

else:
    st.warning("Choisis au moins deux banques.")
