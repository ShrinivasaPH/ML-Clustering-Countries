import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained GMM model
with open("gmm_model.pkl", 'rb') as f:
    model = pickle.load(f)

# Load the same scaler used during training
with open("scaler.pkl", 'rb') as f:
    scaler = pickle.load(f)

# Load dataset for reference
st.title(":blue[Clustering Economies of Countries using Machine Learning]")
df = pd.read_csv('Country-data.csv')
st.dataframe(df.head())
st.caption("Sample view of the dataset.")

st.divider()
st.header(":blue[**Select socio-economic parameters below:**]")

# User input fields
col1, col2 = st.columns(2)

child_mort = col1.number_input("Child Mortality Rate:", 2.6, 142.857)
exports = col2.number_input("Exports (% of GDP):", 0.1090, 92.6750)
health = col1.number_input("Health Spending (% of GDP):", 1.8100, 14.1200)
imports = col2.number_input("Imports (% of GDP):", 0.0659, 101.5750)
income = col1.number_input("Income per Capita:", 609.0000, 51967.5000)
inflation = col2.number_input("Inflation Rate:", -4.2100, 24.1600)
life_expec = col1.number_input("Life Expectancy:", 48.0500, 82.8000)
total_fer = col2.number_input("Fertility Rate:", 1.1500, 7.0075)

less_earners = col1.toggle("Underpaid Workers")
high_child_mort = col2.toggle("High Child Mortality")

# Convert binary toggles to numerical values
less_earners = 1 if less_earners else 0
high_child_mort = 1 if high_child_mort else 0

st.subheader("Click below to see the country's economic cluster.")
if st.button("Predict Country Type"):
    # Prepare input data as a numpy array
    input_data = np.array([[child_mort, exports, health, imports, income, inflation, life_expec, total_fer,
                            less_earners, high_child_mort]])
    
    # Scale the input data using the same scaler
    input_scaled = scaler.transform(input_data)
    
    # Predict the cluster
    pred = model.predict(input_scaled)
    
    # Display the result
    st.header("The Country's Cluster Based on Socio-Economic Parameters:")
    
    st.markdown(f"""
    <div style="background-color:#FFD700; padding:10px; border-radius:10px; 
                text-align:center; font-size:18px; font-weight:bold; color:black; 
                width: 40%; margin: auto;">
        🔮 Predicted Cluster: {pred[0]}
    </div>
    """, unsafe_allow_html=True)
    
    # Cluster meanings
    st.markdown("""
    <div style="border: 2px solid #ccc; padding: 10px; border-radius: 10px; background-color: #f9f9f9;
                width: 50%; margin: auto; text-align: center;">
        <h4 style="color: #333;">🌍 Cluster Meanings 🌍</h4>
        <ul style="list-style-type: none; padding-left: 0; text-align: left; display: inline-block;">
            <li><span style="color: #007bff; font-weight: bold;">0 - Poor Nation</span> 🌱</li>
            <li><span style="color: #ff5733; font-weight: bold;">1 - Developing Nation</span> ⚠️</li>
            <li><span style="color: #28a745; font-weight: bold;">2 - Rich Nation</span> 💰</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style="border: 3px solid #ddd; padding: 10px; background-color: #f9f9f9; border-radius: 5px; font-size: 12px; color: #666;">
    <p style="color:red; font-size: 18px; font-weight: bold;"><strong>Disclaimer:</strong></p> 
    <p>This application is an <strong>academic project</strong> developed for educational purposes only.</p>
    <p>Users should not rely on this application to make any real-world economic, financial, or policy-related decisions.</p>
</div>
""", unsafe_allow_html=True)
