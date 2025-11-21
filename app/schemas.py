from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, date
from typing import Optional, List

class ImageResponse(BaseModel):
    image_id: int
    file_path: str
    main_class: Optional[str] = None
    main_confidence: Optional[float] = None
    criticality: Optional[int] = None  # Критичность
    created_at: datetime
    count_damage: int
    route_id: Optional[int] = None

    model_config = ConfigDict(from_attributes=True)

class RouteCreate(BaseModel):
    name: str

class RouteResponse(BaseModel):
    id: int
    name: str
    created_at: datetime
    image_count: int

    model_config = ConfigDict(from_attributes=True)

class RouteWithImages(BaseModel):
    name: str
    route_id: int
    images: List[ImageResponse]

    model_config = ConfigDict(from_attributes=True)

class ImageCriticalityUpdate(BaseModel):
    criticality: int = Field(..., ge=1, le=5, description="Степень критичности от 1 до 5")

class DetectionInImage(BaseModel):
    id: int
    defect_type: str
    yolo_confidence: Optional[float] = None
    classif_confidence: Optional[float] = None
    roi_path: Optional[str] = None
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    has_repair_request: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DetectionMapPoint(BaseModel):
    """Информация о повреждении для отображения на карте"""
    detection_id: int
    defect_type: str
    gps_latitude: Optional[float] = None
    gps_longitude: Optional[float] = None
    created_at: datetime
    image_id: int

    model_config = ConfigDict(from_attributes=True)

class ImageCardResponse(BaseModel):
    image_id: int
    file_path: str
    main_class: Optional[str] = None
    main_confidence: Optional[float] = None
    criticality: Optional[int] = None
    created_at: datetime
    detections: List[DetectionInImage]

    model_config = ConfigDict(from_attributes=True)

class RepairRequestResponse(BaseModel):
    repair_request_id: int
    detection_id: int
    image_id: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DetectionStatistics(BaseModel):
    date: date
    count: int

    model_config = ConfigDict(from_attributes=True)

class GeneralStatistics(BaseModel):
    total_repair_requests: int
    total_defects: int
    open_repair_requests: int
    closed_repair_requests: int
    completed_repair_requests: int

    model_config = ConfigDict(from_attributes=True)
