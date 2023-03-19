import pathlib
from typing import Optional
from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
from pymongo import MongoClient
from . import ml
import os
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = pathlib.Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent / "models"
PHISHING_MODEL_DIR = MODELS_DIR / "phishing"
MODEL_PATH = PHISHING_MODEL_DIR / "phishing-model.h5"
TOKENIZER_PATH = PHISHING_MODEL_DIR / "phishing-classifer-tokenizer.json"
METADATA_PATH = PHISHING_MODEL_DIR / "phishing-classifer-metadata.json"

class RequestBody(BaseModel):
    text: str
    can_save: Optional[bool]

PHISHING_MODEL = None

@app.on_event("startup")
def on_startup():
    global PHISHING_MODEL, collection
    PHISHING_MODEL = ml.AIModel(
        model_path = MODEL_PATH,
        tokenizer_path = TOKENIZER_PATH,
        metadata_path = METADATA_PATH
    )

    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

async def handle_request_db_entry(request_body, prediction_dict):
    if not request_body.get("can_save"):
        return

    top_prediction = prediction_dict.get("top_prediction")
    
    db_item = {
        "text": request_body.get("text"),
        "label": top_prediction.get("label"),
        "confidence": top_prediction.get("confidence")
    }
    collection.insert_one(db_item)

@app.post("/phishing")
async def phishing_detect(request: Request, body: RequestBody = Body(...)):
    global PHISHING_MODEL, collection

    request_body = body.dict()
    text = request_body.get('text') or 'Hello World'
    can_save = request_body.get('can_save')
    prediction_dict = PHISHING_MODEL.predict_text(text)
    
    await handle_request_db_entry(request_body, prediction_dict)
    return JSONResponse(content={"input": text, "can_save": can_save, "result": prediction_dict}, headers={
        "Access-Control-Allow-Origin": "*"
    })