from fastapi import FastAPI, UploadFile, File
import shutil
import os
from pipeline import VehicleDetectionPipeline

app = FastAPI()
os.makedirs("temp", exist_ok=True)

pipeline = VehicleDetectionPipeline(
    weights_path="./model/YOLO_bdd100k.pt",
    split_size=14,
    num_boxes=2,
    num_classes=13,
    threshold=0.5
)

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    file_path = f"vid_folder/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_path = f"vid_folder/output_{file.filename}"
    pipeline.process_video(file_path, output_path)
    return {"filename": file.filename, "output_file": output_path}
