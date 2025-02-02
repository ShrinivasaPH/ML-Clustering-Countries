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

st.divider() 
#st.header("  ")
st.header(":blue[**Select socio-economic parameters below:**]")
st.caption("Type or use the buttons.")

col1, col2 = st.columns(2)

#slider
child_mort = col1.number_input("Select the child mortality:",
                         0, 208)
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

less_earners = col1.radio("Underpaid workers: (Yes: 1, No: 0)",
                          options=[1, 0], index=1)  # Default to 'No' (0)

less_life_expectancy = col2.radio("Less Life Expectancy: (Yes: 1, No: 0)",
                                  options=[1, 0], index=1)  # Default to 'No' (0)

high_child_mort = col1.radio("High Child Mortality: (Yes: 1, No: 0)",
                             options=[1, 0], index=1)  # Default to 'No' (0)



def reset_inputs():
    for key in st.session_state.keys():
        if key.startswith("input_"):  # Reset only input fields
            st.session_state[key] = 0

st.subheader("Click the below button to see the country's social status.")
if st.button("Predict Country Type"):
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


st.divider()
st.markdown("""
<div style="border: 3px solid #ddd; padding: 10px; background-color: #f9f9f9; border-radius: 5px; font-size: 12px; color: #666;">
    <p style="color:red; font-size: 18px; font-weight: bold;"><strong>Disclaimer:</strong></p> 
    <p>This application is an <strong>academic project</strong> developed for educational purposes only. The data used in this project and the clustering results generated are based on a machine learning model trained on a specific dataset.</p>
    <p><strong>Users should not rely on this application to make any real-world economic, financial, or policy-related decisions.</strong> The results do not represent official classifications, and the accuracy of the predictions is not guaranteed.</p>
    <p>The creator of this application <strong>bears no responsibility</strong> for any consequences, decisions, or actions taken based on the output of this app. Any inconvenience, loss, or harm resulting from the use of this application is entirely at the user's own risk.</p>
    <p>By using this app, you acknowledge and agree to the terms of this disclaimer.</p>
    <p>Additionally, for the sake of impartiality and to avoid any potential offense, country names are intentionally omitted from the clustering results. This is to ensure that no individual, group, or nation feels misrepresented or unfairly categorized based on the results. The intention is purely academic and not to make any political, social, or economic statements.</p>
</div>
""", unsafe_allow_html=True)
