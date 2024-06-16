from functools import wraps
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from ultralytics import YOLO
import os
import time
import shutil
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Allow CORS
origins = [
    "http://localhost:3000",  # React
    "http://localhost:8080",  # Vue.js
    "http://localhost:8000",  # Angular
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_directory = "uploadedFile"  # Store uploaded images in this folder
output_directory = "detectedImages"  # Store detected images in this folder

if not os.path.exists(image_directory):
    os.makedirs(image_directory)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

model = YOLO("models/best.pt")

def rate_limited(max_calls: int, time_frame: int):
    def decorator(func):
        calls = []

        @wraps(func)
        async def wrapper(*args, **kwargs):
            now = time.time()
            calls_in_time_frame = [call for call in calls if call > now - time_frame]
            if len(calls_in_time_frame) >= max_calls:
                raise HTTPException(status_code=429, detail="Rate limit exceeded!")
            calls.append(now)
            return await func(*args, **kwargs)

        return wrapper

    return decorator

def object_detector(filename):
    file_path = os.path.join(image_directory, filename)
    results = model.predict(source=file_path, conf=0.5)
    detections = []

    # Load the original image
    img = cv2.imread(file_path)
    
    for result in results:
        for detection in result.boxes:
            box = detection.xyxy[0].cpu().numpy().astype(int).tolist()  # bounding box coordinates
            score = detection.conf[0].cpu().numpy()  # confidence score in decimal
            label = model.names[int(detection.cls[0])]  # label/class

            # Draw bounding box on the image
            score_percentage = score * 100  # Convert to percentage
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)  # Red color, thicker line
            cv2.putText(img, f"{label} {score_percentage:.2f}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            detections.append({"box": box, "score": float(score), "label": label})  # Convert score to float

    # Save the image with detections
    output_filename = f"{uuid.uuid4()}.jpg"
    output_filepath = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_filepath, img)
    
    return detections, output_filename

def cleanup_old_files(directory, max_files=2):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if len(files) > max_files:
        files.sort(key=os.path.getctime)
        for f in files[:-max_files]:
            os.remove(f)

@app.get("/")
@rate_limited(max_calls=100, time_frame=60)  # Decorator to limit requests
async def index():
    return {"message": "Hello World"}

@app.post("/upload")
@rate_limited(max_calls=100, time_frame=60)  # Decorator to limit requests
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        file.filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(image_directory, file.filename)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        detections, detected_image_filename = object_detector(file.filename)
        os.remove(file_path)  # Clean up the uploaded file

        # Generate the full URL for the detected image
        base_url = str(request.base_url).rstrip("/")
        detected_image_url = f"{base_url}/detectedImages/{detected_image_filename}"

        # Clean up old files in the detectedImages directory
        cleanup_old_files(output_directory)

        # Determine the status of ulat
        status_ulat = "true" if any(d["label"] == "ulat" for d in detections) else "false"

        return JSONResponse(content={"detections": detections, "status_ulat": status_ulat, "photo_detected": detected_image_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files directory
app.mount("/detectedImages", StaticFiles(directory=output_directory), name="detectedImages")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
