import os
import uuid
from datetime import datetime, timedelta
from fastapi import Query
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from ultralytics import YOLO
import cv2

import crud
import database
import utils
import schemas  # Импортируем схемы
import torch
from torchvision import models as tv_models
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo_model = YOLO("../best19final.pt")

classif_model = tv_models.resnet18(pretrained=False)
classif_model.fc = torch.nn.Linear(classif_model.fc.in_features, len(utils.CLASSIF_NAMES))
classif_model.load_state_dict(torch.load("../defect_classifier_final.pth", map_location=utils.device))
classif_model.to(utils.device)
classif_model.eval()

allowed_main_classes = utils.allowed_main_classes


async def get_db():
    async with database.AsyncSessionLocal() as db:
        yield db


@app.post("/predict/")
async def predict(files: List[UploadFile] = File(...), db: AsyncSession = Depends(get_db)):
    all_results = []

    for file in files:
        contents = await file.read()
        img_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img_array is None:
            continue

        unique_project_dir = os.path.join("runs", "detect")
        unique_name = "predict_" + str(uuid.uuid4())

        results = yolo_model.predict(img_array, conf=0.5, save=True, project=unique_project_dir, name=unique_name)

        yolo_img_path = results[0].save_dir + "/" + results[0].path

        index = yolo_img_path.find("runs")
        if index != -1:
            yolo_img_path = yolo_img_path[index:]

        max_confidence = 0.0
        main_class_name = None
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in allowed_main_classes and conf > max_confidence:
                max_confidence = conf
                main_class_name = allowed_main_classes[cls]

        image_record = await crud.create_image(db, yolo_img_path, main_class_name, max_confidence)

        os.makedirs("rois", exist_ok=True)

        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls not in [5, 6]:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            roi = img_array[y1:y2, x1:x2]
            roi_path = utils.save_roi(roi)
            defect_type, confidence = utils.predict_defect(classif_model, roi)

            await crud.create_detection(db, image_record.id, defect_type, conf, confidence, roi_path)

        all_results.append({
            "image_id": image_record.id,
            "file_path": yolo_img_path,
            "main_class": main_class_name,
            "main_confidence": max_confidence,
            "created_at": image_record.created_at  # Добавили created_at
        })

    return JSONResponse(content=[
        {
            "image_id": r["image_id"],
            "file_path": r["file_path"],
            "main_class": r["main_class"],
            "main_confidence": r["main_confidence"],
            "created_at": r["created_at"].isoformat()  # Конвертируем в ISO формат
        }
        for r in all_results
    ])


@app.get("/images/", response_model=List[schemas.ImageResponse])
async def get_all_images(db: AsyncSession = Depends(get_db)):
    images = await crud.get_all_images(db)
    return [
        schemas.ImageResponse(
            image_id=img.id,
            file_path=img.file_path,
            main_class=img.main_class,
            main_confidence=img.main_confidence,
            created_at=img.created_at,
            count_damage=len(img.detections)
        )
        for img in images
    ]


# НОВЫЙ ЭНДПОИНТ: Статистика детекций
@app.get("/statistics/detections", response_model=List[schemas.DetectionStatistics])
async def get_detections_statistics(
        start_date: Optional[str] = Query(None, description="Начальная дата в формате YYYY-MM-DD"),
        end_date: Optional[str] = Query(None, description="Конечная дата в формате YYYY-MM-DD"),
        db: AsyncSession = Depends(get_db)
):
    """
    Получить статистику детекций за период

    Примеры:
    - /statistics/detections?start_date=2025-01-01&end_date=2025-01-31
    - /statistics/detections (последние 30 дней по умолчанию)
    """
    # Если даты не указаны, берём последние 30 дней
    if not end_date:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    if not start_date:
        start_dt = end_dt - timedelta(days=30)
    else:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

    # Добавляем время к датам для полного охвата
    start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    stats = await crud.get_detections_statistics(db, start_dt, end_dt)

    return [
        schemas.DetectionStatistics(
            date=stat.date,
            count=stat.count
        )
        for stat in stats
    ]


@app.get("/images/{image_id}/", response_model=schemas.ImageCardResponse)
async def get_image_card(image_id: int, db: AsyncSession = Depends(get_db)):
    image = await crud.get_image(db, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    detections = await crud.get_detections_by_image(db, image_id)

    return schemas.ImageCardResponse(
        image_id=image.id,
        file_path=image.file_path,
        main_class=image.main_class,
        main_confidence=image.main_confidence,
        created_at=image.created_at,
        detections=[
            schemas.DetectionInImage(
                id=d.id,
                defect_type=d.defect_type,
                yolo_confidence=d.yolo_confidence,
                classif_confidence=d.classif_confidence,
                roi_path=d.roi_path,
                has_repair_request=d.repair_request is not None,
                created_at=d.created_at
            )
            for d in detections
        ]
    )


@app.post("/detections/{detection_id}/repair_request/", response_model=schemas.RepairRequestResponse)
async def create_repair_request(detection_id: int, description: str = "", db: AsyncSession = Depends(get_db)):
    detection = await crud.get_detection(db, detection_id)
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    existing = await crud.get_repair_request_by_detection(db, detection_id)
    if existing:
        raise HTTPException(status_code=400, detail="Repair request already exists for this detection")

    repair_request = await crud.create_repair_request(db, detection_id, description)

    return schemas.RepairRequestResponse(
        repair_request_id=repair_request.id,
        detection_id=detection_id,
        status=repair_request.status,
        description=repair_request.description,
        created_at=repair_request.created_at
    )


@app.get("/detections/{detection_id}/repair_request/", response_model=schemas.RepairRequestResponse)
async def get_repair_request(detection_id: int, db: AsyncSession = Depends(get_db)):
    repair_request = await crud.get_repair_request_by_detection(db, detection_id)
    if not repair_request:
        raise HTTPException(status_code=404, detail="Repair request not found")

    return schemas.RepairRequestResponse(
        repair_request_id=repair_request.id,
        detection_id=repair_request.detection_id,
        status=repair_request.status,
        description=repair_request.description,
        created_at=repair_request.created_at
    )


@app.get("/repair_requests/", response_model=List[schemas.RepairRequestResponse])
async def get_all_repair_requests(db: AsyncSession = Depends(get_db)):
    requests = await crud.get_all_repair_requests(db)
    return [
        schemas.RepairRequestResponse(
            repair_request_id=r.id,
            detection_id=r.detection_id,
            status=r.status,
            description=r.description,
            created_at=r.created_at
        )
        for r in requests
    ]


@app.patch("/repair_requests/{request_id}/")
async def update_repair_request_status(request_id: int, status: str, db: AsyncSession = Depends(get_db)):
    repair_request = await crud.update_repair_request_status(db, request_id, status)
    if not repair_request:
        raise HTTPException(status_code=404, detail="Repair request not found")
    return {
        "repair_request_id": repair_request.id,
        "status": repair_request.status,
        "created_at": repair_request.created_at.isoformat()  # Добавили created_at
    }


@app.get("/statistics/general", response_model=schemas.GeneralStatistics)
async def get_general_statistics(db: AsyncSession = Depends(get_db)):
    """
    Получить общую статистику системы

    Возвращает:
    - total_repair_requests: Общее количество заявок на ремонт
    - total_defects: Общее количество обнаруженных дефектов
    - open_repair_requests: Количество открытых заявок
    - closed_repair_requests: Количество закрытых заявок
    - completed_repair_requests: Количество выполненных заявок
    """
    stats = await crud.get_general_statistics(db)
    return schemas.GeneralStatistics(**stats)

app.mount("/runs", StaticFiles(directory="runs"), name="runs")
app.mount("/rois", StaticFiles(directory="rois"), name="rois")
