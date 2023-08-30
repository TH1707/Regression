import streamlit as st
import os
import pandas as pd
import joblib

# Define the heading style as HTML
heading_style = '''
<div style="color:red;" align='center'>
<h1>Loan Amount Prediction System</h1>
</div>
'''

# Define a function to return the input data as a DataFrame
def return_df(tv, radio, newspaper):
    data = {
        'tv': [tv],
        'radio': [radio],
        'newspaper': [newspaper]
    }
    final_df = pd.DataFrame(data)
    return final_df

# Define a function to load the trained model
def base_model():
    model = joblib.load('finalized_model_regression.pkl')
    return model

# Streamlit UI
st.markdown(heading_style, unsafe_allow_html=True)

# Input fields for advertising expenses
tv = st.number_input('How Much Money Was Spent On TV')
radio = st.number_input('How much money was spent on Radio')
newspaper = st.number_input('How much money was spent on Newspaper')

# Get the input data as a DataFrame
df = return_df(tv, radio, newspaper)

# Check if the "Submit" button is clicked
if st.button('Submit'):
    model = base_model()
    preds = model.predict(df)
    prediction = preds[0]
    st.write("Predicted Loan Amount:", prediction)
