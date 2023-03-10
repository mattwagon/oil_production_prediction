from fastapi import FastAPI

api = FastAPI()

# define a root `/` endpoint
@api.get("/")
def root():
    return {"ok": "API connected"}


@api.get("/predict")
def predict(feature1, feature2):

    # model = picle.load_model()
    # prediction = model.predict(feature1, feature2)

    # Here, I'm only returning the features, since I don't actually have a model.
    # In a real life setting, you would return the predictions.

    return {'prediction': int(feature1)*int(feature2)}

@api.get("/display_data")
def isplay_data(oil_rate):

    # We could return here the display of the real vs. predicted Oil Rates
    # to be discussed together
    return {"data": oil_rate}
