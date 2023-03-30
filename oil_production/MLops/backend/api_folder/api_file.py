from fastapi import FastAPI
import tensorflow as tf
import pandas as pd

api = FastAPI()

model = tf.keras.models.load_model("saved_model")

# define a root `/` endpoint
@api.get("/")
def root():
    return {"ok": "API connected"}


@api.post("/predict")
async def predict(input_data: dict):

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)

    return {"prediction": int(prediction)}

    # Here, I'm only returning the features, since I don't actually have a model.
    # In a real life setting, you would return the predictions.

    # return {'prediction': int(feature1)*int(feature2)}

@api.get("/display_data")
def isplay_data(oil_rate):

    # We could return here the display of the real vs. predicted Oil Rates
    # to be discussed together
    return {"data": oil_rate}
