from fastapi import FastAPI, Header, File, UploadFile, Form
from pymongo.mongo_client import MongoClient
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
from bson import ObjectId
import boto3
from io import BytesIO
import tensorflow as tf
import numpy as np
import jwt
from PIL import Image

app.max_request_size = 1024 * 1024  * 1000
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

s3 = boto3.client('s3', aws_access_key_id='AKIAX6VTAEMG2LGPXFWT', aws_secret_access_key='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')
bucket_name = 'carimageapp'
s3_folder = 'model'
local_model_filename = 'model_backuplamp.h5'

s3.download_file(bucket_name, s3_folder, local_model_filename)

MODEL = tf.keras.models.load_model(local_model_filename)

@app.post("/upload")
async def upload(filesF: List[UploadFile] = File(...),
                 filesR: List[UploadFile] = File(...),
                 filesB: List[UploadFile] = File(...)):

    CLASS_NAMES =  ['Honda Accord 2017' , 'Honda Accord 2018', 'Honda Civic 2017 Fc', 'Honda Civic 2018', 'Honda Civic 2019', 'Honda Civic 2020',
                    'Honda Civic 2021', 'Toyota Camry 2017', 'Toyota Camry 2018', 'Toyota Camry 2019', 'Toyota Camry 2020', 'Toyota Camry 2021',
                    'Toyota CorollaCross 2022', 'Toyota CorollaCross 2023', 'Toyota Yaris 2017 Hatchback', 'Toyota Yaris 2018 ATIV', 'Toyota Yaris 2019 ATIV',  
                    'Toyota Yaris 2020 Hatchback']
    
    images = []

    for file in filesF:
        image = read_file_as_image(await file.read())
        images.append(image)

    img_batches = np.array(images)

    predictions = MODEL.predict(img_batches)

    max_confidences = np.max(predictions, axis=1)

    max_max_confidence = np.max(max_confidences)

    max_confidence_index = np.argmax(max_confidences)
    predicted_class = CLASS_NAMES[max_confidence_index]

    print(max_max_confidence)
    print(max_confidence_index)
    print(predicted_class)

    
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image
