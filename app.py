import streamlit as st
import pandas as pd
import numpy as np

st.title("Portefeuille Markowitz - Banques BVMT")

# données exemple
data = {
    "BIAT": [110, 112, 111, 115, 118],
    "BT": [7, 7.2, 7.1, 7.3, 7.4],
    "AB": [25, 26, 25.5, 26.5, 27]
}

df = pd.DataFrame(data)

st.subheader("Prix des banques")
st.dataframe(df)

# calcul rendements
returns = df.pct_change().dropna()

st.subheader("Rendements")
st.dataframe(returns)

# moyenne
mean_returns = returns.mean()
volatility = returns.std()

st.subheader("Indicateurs")
st.write("Rendements moyens :", mean_returns)
st.write("Volatilité :", volatility)
