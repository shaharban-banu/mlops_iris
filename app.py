from fastapi import FastAPI
import logging
import pickle

logging.basicConfig(filename="app.log",level=logging.INFO)

app=FastAPI()

model=pickle.load(open("model.pkl","rb"))

@app.get('/')
def get_home():
    return {'message':"Iris model API......."}

@app.get('/predict')
def predict(f1:float,f2:float,f3:float,f4:float):
    pred=model.predict([[f1,f2,f3,f4]])
    logging.info(f"input :{[f1,f2,f3,f4]} output : {pred[0]}")
    return {'prediction':int(pred[0])}