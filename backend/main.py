from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from utils import predict, input_trash
import time
import uvicorn
import shutil
import os

PUBLIC_IP_ADDRESS = os.environ.get("PUBLIC_IP_ADDRESS")
PUBLIC_DNS_ADDRESS = os.environ.get("PUBLIC_DNS_ADDRESS")
origins = [ 
    # f"http://{PUBLIC_IP_ADDRESS}:8888",
    # f"http://{PUBLIC_IP_ADDRESS}:8000",
    # f"http://{PUBLIC_DNS_ADDRESS}:8888",
    # f"http://{PUBLIC_DNS_ADDRESS}:8000",
    "http://localhost:8888", # For debugging
    "http://localhost:8000", # For debugging
    "http://localhost:3000", # For debugging
]


app = FastAPI(
    title="🗑️ Garbage Classification",
    summary= "This is an API that helps users classify waste and learn how to handle it"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define the upload directory
UPLOAD_DIR = Path("upload")
UPLOAD_DIR.mkdir(exist_ok=True)

class trash(BaseModel):
    type_trash:  str
    
@app.get("/download/{id}")
async def get_image(id: str):
    try:
        file_path = UPLOAD_DIR / id
        return FileResponse(file_path)
    except FileNotFoundError:
        return {"error": "Image not found"}

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.mkdir(UPLOAD_DIR)

        # Save the uploaded file
        upload_path = f"{UPLOAD_DIR}/{file.filename}"
        with upload_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"path": f"Upload successful: {upload_path}"}
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

@app.get("/")
async def root():
    return {"message" : "This is a garbage classification API",
            "help": "Use /predict to get the output for classification"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # Save the uploaded file
    upload_path = UPLOAD_DIR / file.filename
    with upload_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    predicted_value, predicted_accuracy = await predict(upload_path)

    return {
        "path": upload_path,
        "predicted_value": predicted_value,
        "predicted_accuracy": predicted_accuracy
    }
@app.post("/get-advice")
async def give_advice(Trash: trash ):
    start_time = time.time()
    advice = input_trash(f"Jenis sampah: {Trash.type_trash}")
    return {
        "advice": advice,
        "time": time.time() - start_time
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True
    )
