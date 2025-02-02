import streamlit as st
import pandas as pd
import numpy as np
import pickle


#load the model from disk
with open("gmm_model.pkl", 'rb') as f:
    model = pickle.load(f)


df = pd.read_csv('Country-data.csv')

st.title('Clusters world economies using ML')
st.dataframe(df.head())
st.header("  ")
st.header("Select socio-economic parameters below:")

col1, col2 = st.columns(2)

#slider
child_mort = col1.slider("Select the child mortality:",
                         0, 208, step=5)

exports = col2.slider("Select the export value:",
                      0, 200, step=5)
health = col1.slider("Select the Health-Spending value:",
                         1, 15, step=1)
imports = col2.slider("Select the import value:",
                      0, 175, step=5)
income = col1.slider("Select the income value:",
                      600, 52000, step=10)
inflation = col2.slider("Select the inflation value:",
                      0, 25, step=1)
life_expec = col1.slider("Select the Life Expectency value:",
                         32, 83, step=1)
total_fer = col2.slider("Select the fertility value:",
                      0, 8, step=1)
less_earners = col1.number_input("Underpaid workers:(Yes:1, No:0) ",
                                 0,1,step=1)
less_life_expectancy = col2.number_input("Less Life Expectency.(Yes:1, No:0) ",
                                         0,1,step=1)
high_child_mort = col1.number_input("High Child Mortality (Yes:1, No:0)",
                                    0,1,step=1)

if st.button("Country Type"):
    col1, col2 = st.columns(2)

    
    input_data = np.array([5, 45, 3, 54, 3500, 20, 70, 3, less_earners, less_life_expectancy, high_child_mort]).reshape(1, -1)

    pred = model.predict(input_data)

    st.write("Your Country's Cluster based on your selected parameters: ", pred)


    st.write("")
    st.write("The Selected values:")
    st.write("Child import level:",child_mort)
    st.write("Export Levels:",exports)