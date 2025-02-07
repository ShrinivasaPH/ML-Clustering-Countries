import streamlit as st
import pandas as pd
import numpy as np
import pickle


#load the model from disk
with open("gmm_model.pkl", 'rb') as f:
    model = pickle.load(f)

scaler = model.scaler

df = pd.read_csv('Country-data.csv')
st.title(":blue[Clustering Economies of Countries using Machine Learning]")
st.dataframe(df.head())
st.caption("Sample view of the dataset.")

st.divider() 
#st.header("  ")
st.header(":blue[**Select socio-economic parameters below:**]")
st.caption("Type or use the buttons.")

col1, col2 = st.columns(2)

#slider
child_mort = col1.number_input("Select the Child Mortality Rate:",
                         2.6, 142.857)
exports = col2.number_input("Select the Export value:",
                      0.1090, 92.6750)
health = col1.number_input("Select the Health-Spending Rate:",
                         1.8100, 14.1200)
imports = col2.number_input("Select the Import Rate:",
                      0.0659, 101.5750)
income = col1.number_input("Select the Income value:",
                      609.0000, 51967.5000)
inflation = col2.number_input("Select the Inflation Rate:",
                      -4.2100, 24.1600)
life_expec = col1.number_input("Select the Life Expectency Rate:",
                         48.0500, 82.8000)
total_fer = col2.number_input("Select the Fertility Rate:",
                      1.1500, 7.0075)

less_earners = col1.toggle("Underpaid workers:")
                          
#less_life_expectancy = col2.toggle("Less Life Expectancy: (Yes: 1, No: 0)")

high_child_mort = col2.toggle("High Child Mortality: ")
                   

#less_earners = 1
#less_life_expectancy = 1
#high_child_mort = 1

def reset_inputs():
    for key in st.session_state.keys():
        if key.startswith("input_"):  # Reset only input fields
            st.session_state[key] = 0

st.subheader("Click the below button to see the country's social status.")
if st.button("Predict Country Type"):
    col1, col2 = st.columns(2)

    input_data = np.array([child_mort, exports, health, imports, income, inflation, life_expec, total_fer,
                            less_earners, high_child_mort]).reshape(1, -1)

    pred = model.predict(input_data)

    st.header("The Country's Cluster based on the selected socio-economic parameters:")

    st.markdown(
    f"""
    <div style="background-color:#FFD700; padding:10px; border-radius:10px; 
                text-align:center; font-size:18px; font-weight:bold; color:black; 
                width: 40%; margin: auto;">
        🔮 Predicted Cluster: {pred[0]}
    </div>
    """, unsafe_allow_html=True
)



    st.write(" ")

    st.markdown(
    """
    <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9;
                width: 50%; margin: auto; text-align: center;">
        <h4 style="color: #333;">🌍 Cluster Meanings 🌍</h4>
        <ul style="list-style-type: none; padding-left: 0; text-align: left; display: inline-block;">
            <li><span style="color: #007bff; font-weight: bold;">0 - Poor Nation</span> 🌱</li>
            <li><span style="color: #ff5733; font-weight: bold;">1 - Developing Nation</span> ⚠️</li>
            <li><span style="color: #28a745; font-weight: bold;">2 - Rich Nation</span> 💰</li>
        </ul>
    </div>
    """, 
    unsafe_allow_html=True
)


   
   

    


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
