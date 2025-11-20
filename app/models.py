from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, func
from sqlalchemy.orm import relationship
from database import Base

class ImageRecord(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_path = Column(String, unique=True, index=True)
    main_class = Column(String, nullable=True)
    main_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    detections = relationship("Detection", back_populates="image")

class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id"), nullable=False)
    defect_type = Column(String, nullable=False)
    yolo_confidence = Column(Float)
    classif_confidence = Column(Float)
    roi_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    image = relationship("ImageRecord", back_populates="detections")
    repair_request = relationship("RepairRequest", back_populates="detection", uselist=False)

class RepairRequest(Base):
    __tablename__ = "repair_requests"
    id = Column(Integer, primary_key=True, index=True)
    detection_id = Column(Integer, ForeignKey("detections.id"), unique=True, nullable=False)
    status = Column(String, default="open")
    description = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    detection = relationship("Detection", back_populates="repair_request")
