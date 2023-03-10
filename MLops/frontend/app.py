import streamlit as st
import requests

import numpy as np
import pandas as pd

#st.set_page_config(
            #page_title="Oil Production Prediction", # => Quick reference - Streamlit
            #page_icon="🐍",
            #layout="centered", # wide
            #initial_sidebar_state="auto") # collapsed

# Style the page with CSS in we have some time ?

st.markdown("""# Oil Production Prediction
### Predict the **Oil Rate** based on various parameters:
- a
- b
- c
---
""")

from PIL import Image
image = Image.open('Image_OPP_big.png')
st.image(image, caption='Oil, Water and Gas Production rates in one of the Wells', use_column_width=False)

st.checkbox("Well A")
st.checkbox("Well B")
st.checkbox("Well C")

param1 = st.slider('Select a number', 1, 10, 3)

param2 = st.slider('Select another number', 1, 10, 3)

#param2 = st.number_input('Insert another number')
#st.write('The current number is ', number)

#url = 'http://localhost:8080/predict'
url = 'https://container-opp-6kchhmx67a-ew.a.run.app/predict'

params = {
    'feature1': param1,  # 0 for Sunday, 1 for Monday, ...
    'feature2': param2
}
response = requests.get(url, params=params)

st.text(response.json())

# Here will be the Oil Production plot (line chart) with the predicted values
# and how they are compared to the real values
@st.cache
def get_line_chart_data():

    return pd.DataFrame(
            np.random.randn(20, 3),
            columns=['a', 'b', 'c']
        )

df = get_line_chart_data()

st.line_chart(df)
