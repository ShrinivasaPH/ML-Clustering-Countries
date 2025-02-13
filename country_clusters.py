import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model from disk
with open("gmm_model.pkl", 'rb') as f:
    model = pickle.load(f)
scaler = model.scaler

# Define the mapping for cluster labels
cluster_mapping = {0: "Underdeveloped Nation", 1: "Developed Nation", 2: "Developing Nation"}

st.title(":blue[Clustering the World-Economy using Machine Learning]")
st.divider()
st.image("Thumbnail.jpeg", caption="Clustering")
st.divider()

# How to Use Section
st.subheader("📌 How to Use This App")
st.markdown(
    """
    1. **Select socio-economic parameters** using the input fields below.
    2. You can either type in values manually or use the step buttons.
    3. Click the **'Predict Country Type'** button to determine the economic cluster.
    4. The model will categorize the input into one of the following clusters:
       - 🟥 **Underdeveloped Nation**
       - 🟩 **Developed Nation**
       - 🟦 **Developing Nation**
    5. The prediction is displayed in a highlighted box below the button.
    6. ⚠️ **Note:** This tool is for educational purposes only and should not be used for real-world decision-making.
    """
)

st.header(":blue[**Select socio-economic parameters below:**]")
st.caption("Type or use the buttons.")

col1, col2 = st.columns(2)

# Slider inputs for the user
child_mort = col1.number_input("Child Mortality Rate:", 2.6, 208.000000)
exports = col2.number_input("Export value:", 0.1090, 200.000000)
health = col1.number_input("Health-Spending Rate:", 1.8100, 17.900000)
imports = col2.number_input("Import Rate:", 0.0659, 174.000000)
income = col1.number_input("Income value:", 609.0000, 125000.000000)
inflation = col2.number_input("Inflation Rate:", -4.2100, 104.000000)
life_expec = col1.number_input("Life Expectancy Rate:", 32.100000, 82.800000)
total_fer = col2.number_input("Fertility Rate:", 1.1500, 7.490000)
gdpp = col1.number_input("GDPP Rate:", 231.000000, 105000.000000)
net_export_ratio = col2.number_input("Net Export Ratio:", -0.224771, 0.011095)

less_earners = col1.checkbox("Less Earners:")
high_child_mort = col2.checkbox("High Child Mortality:")

# Prepare user data for prediction
user_data = pd.DataFrame([
    [child_mort, exports, health, imports, income, inflation, life_expec, total_fer,
    gdpp, net_export_ratio, less_earners, high_child_mort]
], columns=['child_mort', 'exports', 'health', 'imports', 'income', 'inflation', 'life_expec',
            'total_fer', 'gdpp', 'net_export_ratio', 'less_earners', 'high_child_mort'])

# Scale the user data
user_data_scaled = scaler.transform(user_data)

st.subheader("Click the below button to see the country's social status.")
if st.button("Predict Country Type"):
    # Predict the cluster based on the scaled user data
    pred = model.predict(user_data_scaled)[0]
    pred_label = cluster_mapping.get(pred, "Unknown")  # Get corresponding label

    st.header("The Country's Cluster based on the selected socio-economic parameters:")

    st.markdown(
        f"""
        <div style="background-color:#FFD700; padding:10px; border-radius:10px; 
                    text-align:center; font-size:18px; font-weight:bold; color:black; 
                    width: 40%; margin: auto;">
            🔮 Predicted Cluster: <br> {pred_label}
        </div>
        """, unsafe_allow_html=True
    )

    st.write(" ")

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
