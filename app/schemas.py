from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from datetime import datetime, date



class ImageResponse(BaseModel):
    image_id: int
    file_path: str
    main_class: Optional[str] = None
    main_confidence: Optional[float] = None
    created_at: datetime
    count_damage: int

    model_config = ConfigDict(from_attributes=True)


class DetectionInImage(BaseModel):
    id: int
    defect_type: str
    yolo_confidence: Optional[float] = None
    classif_confidence: Optional[float] = None
    roi_path: Optional[str] = None
    has_repair_request: bool
    created_at: datetime


class ImageCardResponse(BaseModel):
    image_id: int
    file_path: str
    main_class: Optional[str] = None
    main_confidence: Optional[float] = None
    created_at: datetime
    detections: List[DetectionInImage]


class RepairRequestResponse(BaseModel):
    repair_request_id: int
    detection_id: int
    status: str
    description: Optional[str] = None
    created_at: datetime

# Новая схема для статистики
class DetectionStatistics(BaseModel):
    date: date
    count: int


class GeneralStatistics(BaseModel):
    total_repair_requests: int
    total_defects: int
    open_repair_requests: int
    closed_repair_requests: int
    completed_repair_requests: int


