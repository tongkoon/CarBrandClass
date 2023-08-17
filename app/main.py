import base64
import pickle

import cv2
import numpy as np
import requests
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# use docker
from app.code import predict_brand

# from code import predict_brand

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)
# model = pickle.load(open(r'C:\Users\User\Desktop\AI\Assignment1\CarBrandClass\model\model_XGB.pkl','rb'))
# use docker
model = pickle.load(open(f'model/model_XGB.pkl','rb'))

# end_hog = 'http://localhost:8080/api/gethog'
# use docker
end_hog = 'http://172.17.0.2:80/api/gethog'

@app.get("/")
def root():
    return {"message": "This is my api"}

@app.post("/api/carbrand")
async def read_str(request:Request):
    item = await request.json()
    hog = requests.get(end_hog,json=item)
    res = predict_brand(model,hog.json()['HOG'])
    return res
