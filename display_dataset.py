import streamlit as st
import pandas as pd
import numpy as np


def display_dataset():
    st.title("Country Clusters")
    
    df = pd.read_csv('Country-data.csv')
    st.title(":blue[Clustering Economies of Countries using Machine Learning]")
    st.dataframe(df.head())
    st.caption("Sample view of the dataset.")

# Call the function directly
display_dataset()