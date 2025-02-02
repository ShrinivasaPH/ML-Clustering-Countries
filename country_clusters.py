import streamlit as st
import pandas as pd
import numpy as np
import pickle


#load the model from disk
with open("gmm_model.pkl", 'rb') as f:
    model = pickle.load(f)


df = pd.read_csv('Country-data.csv')

st.title(":blue[Clustering Economies of Countries using Machine Learning]")
st.dataframe(df.head())
#st.header("  ")
st.header(":blue[**Select socio-economic parameters below:**]")

col1, col2 = st.columns(2)

#slider
child_mort = col1.number_input("Select the child mortality:",
                         0, 208, step=1)
exports = col2.number_input("Select the export value:",
                      0, 200, step=1)
health = col1.number_input("Select the Health-Spending value:",
                         0, 15, step=1)
imports = col2.number_input("Select the import value:",
                      0, 175, step=1)
income = col1.number_input("Select the income value:",
                      600, 52000, step=10)
inflation = col2.number_input("Select the inflation value:",
                      0, 25, step=1)
life_expec = col1.number_input("Select the Life Expectency value:",
                         32, 83, step=1)
total_fer = col2.number_input("Select the fertility value:",
                      0, 8, step=1)
less_earners = col1.slider("Underpaid workers:(Yes:1, No:0) ",
                                 0,1,step=1)
less_life_expectancy = col2.slider("Less Life Expectency.(Yes:1, No:0) ",
                                         0,1,step=1)
high_child_mort = col1.slider("High Child Mortality (Yes:1, No:0)",
                                    0,1,step=1)


def reset_inputs():
    for key in st.session_state.keys():
        if key.startswith("input_"):  # Reset only input fields
            st.session_state[key] = 0

if st.button("Country Type"):
    col1, col2 = st.columns(2)

    input_data = np.array([5, 45, 3, 54, 3500, 20, 70, 3, less_earners, less_life_expectancy, high_child_mort]).reshape(1, -1)

    pred = model.predict(input_data)

#if st.button("Reset"):
#    reset_inputs()
#    st.rerun()

    st.title("Your Country's Cluster based on your selected parameters:")
    st.header(pred)
    st.subheader(" ", divider="gray")
    st.write("Developing Nation:0") 
    st.write("Poor Nation:1")
    st.write("Rich Nation:2")

if st.button("Refresh"):
    st.rerun()  # Completely reloads the app
