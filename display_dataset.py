import streamlit as st
import pandas as pd
import numpy as np

def dataset():
    st.title("DataSet")

pg = st.navigation([st.Page("country_clusters.py"), st.Page(dataset)])
pg.run()