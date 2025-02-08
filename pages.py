import streamlit as st

st.page_link("country_clusters.py", label="Home", icon="🏠")
st.page_link("pages/display_dataset.py", label="Page 1", icon="📊")
st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
st.page_link("http://www.google.com", label="Google", icon="🌎")