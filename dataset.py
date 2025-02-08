import streamlit as st

def dataset():
    st.title("DataSet")

pg = st.navigation([st.Page("country_clusters.py"), st.Page(dataset)])
pg.run()