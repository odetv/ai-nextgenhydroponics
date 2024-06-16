from functools import wraps
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from ultralytics import YOLO
import os
import shutil
import cv2
import numpy as np
from fastapi.staticfiles import StaticFiles
from io import BytesIO
import requests
import base64

app = FastAPI()

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

def object_detector(source):
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source)
        if response.status_code == 200:
            # Baca gambar dari BytesIO
            image_np = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        else:
            raise ValueError(f"Gagal mengambil gambar dari URL: {source}")
    elif source.startswith("data:image"):
        # Handle base64 encoded image
        base64_str = source.split(",")[1]
        # Decode base64 string
        image_data = base64.b64decode(base64_str)
        # Read decoded image data using OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        # Diasumsikan source adalah jalur file lokal
        img = cv2.imread(source)

    results = model.predict(source=img, conf=0.5)

    detections = []

    for result in results:
        for detection in result.boxes:
            box = detection.xyxy[0].cpu().numpy().astype(int).tolist()  # koordinat kotak pembatas
            score = detection.conf[0].cpu().numpy()  # skor kepercayaan dalam desimal
            label = model.names[int(detection.cls[0])]  # label/kelas
            score_percentage = score * 100  # Ubah menjadi persentase
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)  # Warna merah, garis lebih tebal
            cv2.putText(img, f"{label} {score_percentage:.2f}%", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            detections.append({"box": box, "score": float(score), "label": label})  # Konversi skor menjadi float
            
    output_filename = f"{uuid.uuid4()}.jpg"
    output_filepath = os.path.join(output_directory, output_filename)
    cv2.imwrite(output_filepath, img)

    return detections, output_filename

def cleanup_old_files(directory, max_files=50):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if len(files) > max_files:
        files.sort(key=os.path.getctime)
        for f in files[:-max_files]:
            os.remove(f)

@app.get("/")
async def index():
    return {"message": "API Model AI Next-Gen Hydroponics"}

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(None), image_url: str = Form(None)):
    try:
        if file:
            # Handle UploadFile
            file.filename = f"{uuid.uuid4()}.jpg"
            file_path = os.path.join(image_directory, file.filename)

            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
        elif image_url:
            # Handle URL
            if image_url.startswith("data:image"):
                # Handle base64 encoded image
                # Extract base64 string
                base64_str = image_url.split(",")[1]
                # Decode base64 string
                image_data = base64.b64decode(base64_str)
                # Save decoded image data to a temporary file
                temp_filename = f"{uuid.uuid4()}.jpg"
                temp_file_path = os.path.join(image_directory, temp_filename)

                with open(temp_file_path, "wb") as f:
                    f.write(image_data)

                file_path = temp_file_path
            else:
                # Handle regular URL
                response = requests.get(image_url)
                if response.status_code == 200:
                    # Save image from URL to a temporary file
                    file_extension = image_url.split('.')[-1]
                    temp_filename = f"{uuid.uuid4()}.{file_extension}"
                    temp_file_path = os.path.join(image_directory, temp_filename)

                    with open(temp_file_path, 'wb') as f:
                        f.write(response.content)

                    file_path = temp_file_path
                else:
                    raise HTTPException(status_code=400, detail="Gagal mengambil file dari URL")
        else:
            raise HTTPException(status_code=400, detail="Tidak ada file atau URL yang diberikan")

        detections, detected_image_filename = object_detector(file_path)
        # os.remove(file_path)  # Hapus file yang diunggah/sementara
        cleanup_old_files(image_directory)

        # Generate full URL untuk gambar yang terdeteksi
        base_url = str(request.base_url).rstrip("/")
        detected_image_url = f"{base_url}/detectedImages/{detected_image_filename}"

        # Bersihkan file lama di direktori detectedImages
        cleanup_old_files(output_directory)

        # Tentukan status ulat
        status_ulat = "true" if any(d["label"] == "ulat" for d in detections) else "false"

        return JSONResponse(content={"detections": detections, "status_ulat": status_ulat, "photo_detected": detected_image_url})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.mount("/detectedImages", StaticFiles(directory=output_directory), name="detectedImages")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
