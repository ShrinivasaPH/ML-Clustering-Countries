import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv('Country-data.csv')
st.title(":blue[Clustering Economies of Countries using Machine Learning]")
st.dataframe(df.head())
st.caption("Sample view of the dataset.")