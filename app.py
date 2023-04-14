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

# @app.get("/predict")
# def predict(data1: int, data2: int, data3: int, data4: int, data5: int, data6: int,
#             data7: int, data8: int, data9: int, data10: int, data11: int, data12: int,
#             data13: int, data14: int, data15: int):
#     # Load model from .pkl file
#     with open('./dc_model.pkl','rb') as file:
#         model = pickle.load(file)
#         # Convert input data to DataFrame
#         df = pd.DataFrame({'data1': [data1], 'data2': [data2], 'data3': [data3],
#                             'data4': [data4], 'data5': [data5], 'data6': [data6],
#                               'data7': [data7], 'data8': [data8], 'data9': [data9],
#                                'data10': [data10], 'data11': [data11], 'data12': [data12],
#                                  'data13': [data13], 'data14': [data14], 'data15': [data15]})
#         # Make prediction
#         prediction = model.predict(df)
#         # Return Prediction as JSON response
#         return {'prediction': prediction[0]}

# Define endpoint for making predictions
@app.post('/predict/post')
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


    #     Sample Json Data
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