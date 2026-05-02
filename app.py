import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

st.set_page_config(page_title="Dashboard Markowitz BVMT", layout="wide")

st.title("📊 Dashboard Markowitz - BVMT")
st.write("Analyse financière et optimisation de portefeuille selon Markowitz.")

uploaded_files = st.file_uploader(
    "Importer les fichiers Excel BVMT 2021–2025",
    type=["xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Importe les fichiers Excel pour commencer.")
    st.stop()

all_data = []

for file in uploaded_files:
    df = pd.read_excel(file)
    df["Fichier"] = file.name
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

st.subheader("Aperçu des données importées")
st.dataframe(data.head())

date_col = st.selectbox("Choisir la colonne Date", data.columns)
name_col = st.selectbox("Choisir la colonne Banque / Société", data.columns)
price_col = st.selectbox("Choisir la colonne Cours de clôture", data.columns)

data = data[[date_col, name_col, price_col]].copy()
data.columns = ["Date", "Societe", "Close"]

data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=True)
data["Societe"] = data["Societe"].astype(str).str.strip()

data["Close"] = (
    data["Close"]
    .astype(str)
    .str.replace(",", ".", regex=False)
)

data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
data = data.dropna()
data = data[data["Close"] > 0]

prices = data.pivot_table(
    index="Date",
    columns="Societe",
    values="Close",
    aggfunc="last"
)

prices = prices.sort_index()
prices = prices.ffill().dropna(how="all")

st.subheader("Tableau des prix")
st.dataframe(prices)

societes = list(prices.columns)

selected = st.multiselect(
    "Choisir les banques / sociétés",
    options=societes,
    default=societes[:min(5, len(societes))]
)

rf = st.number_input("Taux sans risque annuel (%)", value=7.5) / 100

if len(selected) < 2:
    st.warning("Choisis au moins deux banques ou sociétés.")
    st.stop()

selected_prices = prices[selected].dropna()
returns = selected_prices.pct_change().dropna()

mean_returns = returns.mean() * 252
volatility = returns.std() * np.sqrt(252)
cov_matrix = returns.cov() * 252
corr_matrix = returns.corr()

st.subheader("📈 Courbes des prix")

prices_long = selected_prices.reset_index().melt(
    id_vars="Date",
    var_name="Societe",
    value_name="Prix"
)

fig_prices = px.line(
    prices_long,
    x="Date",
    y="Prix",
    color="Societe",
    title="Évolution des prix"
)

st.plotly_chart(fig_prices, use_container_width=True)

st.subheader("📈 Rentabilité cumulée")

cumulative_returns = (1 + returns).cumprod() - 1

cum_long = cumulative_returns.reset_index().melt(
    id_vars="Date",
    var_name="Societe",
    value_name="Rentabilité cumulée"
)

fig_cum = px.line(
    cum_long,
    x="Date",
    y="Rentabilité cumulée",
    color="Societe",
    title="Rentabilité cumulée"
)

fig_cum.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_cum, use_container_width=True)

st.subheader("📊 Indicateurs par banque")

metrics = pd.DataFrame({
    "Rentabilité annualisée": mean_returns,
    "Volatilité annualisée": volatility,
    "Sharpe individuel": (mean_returns - rf) / volatility
})

st.dataframe(metrics.style.format("{:.2%}"))

st.subheader("📊 Matrice variance-covariance")
st.dataframe(cov_matrix)

st.subheader("📊 Matrice de corrélation")
st.dataframe(corr_matrix)

n = len(selected)
init = np.ones(n) / n
bounds = tuple((0, 1) for _ in range(n))
constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

def port_return(w):
    return np.dot(w, mean_returns)

def port_vol(w):
    return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))

def neg_sharpe(w):
    vol = port_vol(w)
    if vol == 0:
        return 999
    return -(port_return(w) - rf) / vol

def min_vol(w):
    return port_vol(w)

result_sharpe = minimize(
    neg_sharpe,
    init,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

result_minvar = minimize(
    min_vol,
    init,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints
)

weights_sharpe = result_sharpe.x
weights_minvar = result_minvar.x

ret_sharpe = port_return(weights_sharpe)
vol_sharpe = port_vol(weights_sharpe)
sharpe_ratio = (ret_sharpe - rf) / vol_sharpe

ret_minvar = port_return(weights_minvar)
vol_minvar = port_vol(weights_minvar)
sharpe_minvar = (ret_minvar - rf) / vol_minvar

st.subheader("🚀 Portefeuille optimal - Ratio de Sharpe maximal")

weights_df = pd.DataFrame({
    "Banque / Société": selected,
    "Poids Sharpe": weights_sharpe,
    "Poids variance minimale": weights_minvar
})

st.dataframe(weights_df.style.format({
    "Poids Sharpe": "{:.2%}",
    "Poids variance minimale": "{:.2%}"
}))

col1, col2, col3 = st.columns(3)

col1.metric("Rentabilité Sharpe", f"{ret_sharpe * 100:.2f}%")
col2.metric("Risque Sharpe", f"{vol_sharpe * 100:.2f}%")
col3.metric("Ratio Sharpe", f"{sharpe_ratio:.4f}")

st.subheader("🛡️ Portefeuille à variance minimale")

col4, col5, col6 = st.columns(3)

col4.metric("Rentabilité Min Var", f"{ret_minvar * 100:.2f}%")
col5.metric("Risque Min Var", f"{vol_minvar * 100:.2f}%")
col6.metric("Sharpe Min Var", f"{sharpe_minvar:.4f}")

st.subheader("📌 Graphique des poids")

weights_long = weights_df.melt(
    id_vars="Banque / Société",
    var_name="Portefeuille",
    value_name="Poids"
)

fig_weights = px.bar(
    weights_long,
    x="Banque / Société",
    y="Poids",
    color="Portefeuille",
    barmode="group",
    title="Poids optimaux"
)

fig_weights.update_yaxes(tickformat=".0%")
st.plotly_chart(fig_weights, use_container_width=True)

st.subheader("🌐 Frontière efficiente")

frontier = []

target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 50)

for target in target_returns:
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w, target=target: port_return(w) - target}
    )

    result = minimize(
        min_vol,
        init,
        method="SLSQP",
        bounds=bounds,
        constraints=cons
    )

    if result.success:
        w = result.x
        frontier.append({
            "Risque": port_vol(w),
            "Rentabilité": port_return(w)
        })

frontier_df = pd.DataFrame(frontier)

fig_frontier = go.Figure()

fig_frontier.add_trace(go.Scatter(
    x=frontier_df["Risque"],
    y=frontier_df["Rentabilité"],
    mode="lines",
    name="Frontière efficiente"
))

fig_frontier.add_trace(go.Scatter(
    x=[vol_sharpe],
    y=[ret_sharpe],
    mode="markers",
    name="Portefeuille Sharpe max",
    marker=dict(size=14)
))

fig_frontier.add_trace(go.Scatter(
    x=[vol_minvar],
    y=[ret_minvar],
    mode="markers",
    name="Portefeuille variance minimale",
    marker=dict(size=14)
))

fig_frontier.update_layout(
    title="Frontière efficiente de Markowitz",
    xaxis_title="Risque / Volatilité",
    yaxis_title="Rentabilité"
)

fig_frontier.update_xaxes(tickformat=".0%")
fig_frontier.update_yaxes(tickformat=".0%")

st.plotly_chart(fig_frontier, use_container_width=True)
