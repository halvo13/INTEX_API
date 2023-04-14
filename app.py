import pickle
import pandas as pd
from sklearn import preprocessing
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to Jake's API!"}

# Define endpoint for making predictions
@app.post('/predict')
def predict(data:dict):
  # Load model from .pkl file
  with open('./dc_model.pkl','rb') as file:
    model = pickle.load(file)
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Make prediction
    prediction = model.predict(df)
    # Return Prediction as JSON response
    return {'prediction': prediction[0]}


        # Sample Json Data
    #  {
    #     "squarenorthsouth": 200,
    #     "squareeastwest": 20,
    #     "yellow":0,
    #     "white": 0,
    #     "red":0,
    #     "purple": 0,
    #     "color":0,
    #     "adultsubadult_A": 1,
    #     "adultsubadult_C":0,
    #     "area_NE": 0,
    #     "area_SE":1,
    #     "area_SW": 0,
    #     "depthlevel_Mid_Deep":1,
    #     "depthlevel_Mid_Shallow":0,
    #     "depthlevel_Shallow":0
    # }