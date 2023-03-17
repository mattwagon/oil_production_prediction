import streamlit as st
import requests

import numpy as np
import pandas as pd
import joblib

#st.set_page_config(
            #page_title="Oil Production Prediction", # => Quick reference - Streamlit
            #page_icon="🐍",
            #layout="centered", # wide
            #initial_sidebar_state="auto") # collapsed

# Style the page with CSS in we have some time ?

st.header("""Oil Production Prediction
### Predict the **Oil Rate** based on different parameters:
- Well head temperature and pressure
- Amount of water and gas in production
- Various pressures
---
""")

from PIL import Image
image = Image.open('Image_OPP_big.png')
st.image(image, caption='Oil, Water and Gas Production rates in one of the Wells', use_column_width=False)

st.markdown("""
#### *Please select the well*:
""")
st.checkbox("Well A")
st.checkbox("Well B")

#param1 = st.slider('Select a number', 1, 10, 3)
#param2 = st.slider('Select another number', 1, 10, 3)
#params = {
#    'feature1': param1,
#    'feature2': param2
#}


baseline_model = joblib.load(open("baseline_model.pkl","rb"))

st.markdown("""
#### *Please select the parameters*:
""")
with st.form(key='params_for_api'):

    WHT = st.slider("Choose the Well Head Temperature", 0.0, 1.0, 0.05)
    WHP = st.slider("Choose the Well Head Pressure", 0.0 , 0.6, 0.05)
    Tubing_Gradient = st.slider("Choose the Tubing Gradient", 0.25 , 1.05, 0.05)
    Service_Line_P = st.slider("Choose the Line Pressure", -0.05 , 0.65, 0.05)
    Sand_Raw = st.slider("Choose the Sand Content", -0.2 , 0.8, 0.05)
    Qgas_MPFM = st.slider("Choose the Gas Rate", 0.0 , 0.15, 0.01)
    MPFM_WCT = st.slider("Choose the amount of Water in Production", 0.0 , 1.0, 0.05)
    MPFM_Venturi_dP = st.slider("Choose the Pressure Difference", 0.0 , 0.4, 0.05)
    MPFM_T = st.slider("Choose the Sensor Temperature", 0.0 , 1.0, 0.05)
    Manifold_T = st.slider("Choose the Manifold Temperature", 0.1 , 1.0, 0.05)
    Manifold_P = st.slider("Choose the Manifold Pressure", 0.2 , 1.1, 0.05)
    Jumper_T = st.slider("Choose the Jumper Temperature", 0.1 , 1.0, 0.05)
    Choke_Opening = st.slider("Choose the Choke Size", 0.1 , 0.9, 0.05)
    Annulus_P = st.slider("Choose the Annulus Pressure", 0.2 , 0.95, 0.05)

    st.form_submit_button('Make prediction')

dic = {
    "Well Head Temperature": WHT,
    "Well Head Pressure": WHP,
    "Tubing Gradient": Tubing_Gradient,
    "Service Line Pressure": Service_Line_P,
    "Sand Content": Sand_Raw,
    "Gas Rate": Qgas_MPFM,
    "Amount of Water in Production": MPFM_WCT,
    "Pressure Difference": MPFM_Venturi_dP,
    "Sensor Temperature": MPFM_T,
    "Manifold Temperature": Manifold_T,
    "Manifold Pressure": Manifold_P,
    "Jumper Temperature": Jumper_T,
    "Choke size": Choke_Opening,
    "Annulus Pressure": Annulus_P
}

#url = 'http://localhost:8080/predict'
url = 'https://container-opp-6kchhmx67a-ew.a.run.app/predict'
#response = requests.get(url, params=dic)
#prediction = response.json()
#st.text(response.json())

df_test = pd.DataFrame.from_dict([dic])
prediction = baseline_model.predict(df_test)

#pred = prediction['Qoil MPFM']

st.header(f'The estimated Oil Rate is: {int(prediction)} barrels')

# We can also display the DF if needed:
#st.dataframe(df_test)

# Here will be the Oil Production plot (line chart) with the predicted values
# and how they are compared to the real values
@st.cache
def get_line_chart_data():

    df = pd.read_csv("prediction.csv")
    return df

df = get_line_chart_data()
st.line_chart(df[['Qoil MPFM', 'predicted_Qoil']])
