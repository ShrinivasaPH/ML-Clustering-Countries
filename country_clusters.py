import streamlit as st
import pandas as pd

df = pd.read_csv('Country-data.csv')

st.title('Clusters world economies using ML')
st.dataframe(df.head())

col1, col2 = st.columns(2)

#slider
child_mort = col1.slider("Select the child mortality:",
                         2, 208, step=5)

exports = col2.slider("Select the export value:",
                      0, 200, 5)
imports = col1.slider("Select the import value:",
                      0, 175, 5)
health = col2.slider("Select the Health-Spending value:",
                         1, 15, step=1)
income = col1.slider("Select the income value:",
                      600, 52000, 10)
inflation = col2.slider("Select the inflation value:",
                      -5, 25, 5)
life_expec = col1.slider("Select the Life Expectency value:",
                         32, 83, step=1)
total_fer = col2.slider("Select the fertility value:",
                      1, 8, 1)

if st.button("Country Type")