from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from sqlalchemy import func, cast, Date
from datetime import datetime, timedelta
from typing import Optional, List
import models
from sqlalchemy.orm import joinedload

# Создание маршрута
async def create_route(db: AsyncSession, name: str):
    route = models.Route(name=name)
    db.add(route)
    await db.commit()
    await db.refresh(route)
    return route

# Получить все маршруты с количеством изображений
async def get_all_routes(db: AsyncSession):
    stmt = (
        select(models.Route, func.count(models.ImageRecord.id).label("image_count"))
        .outerjoin(models.Route.images)
        .group_by(models.Route.id)
        .order_by(models.Route.created_at.desc())
    )
    result = await db.execute(stmt)
    return result.all()

# Создать изображение с указанием route_id
async def create_image(
        db: AsyncSession,
        file_path: str,
        main_class: str,
        main_confidence: float,
        route_id: Optional[int] = None
):
    image = models.ImageRecord(
        file_path=file_path,
        main_class=main_class,
        main_confidence=main_confidence,
        route_id=route_id
    )
    db.add(image)
    await db.commit()
    await db.refresh(image)
    return image

# Получить все изображения по route_id
async def get_images_by_route(db: AsyncSession, route_id: int):
    stmt = (
        select(models.ImageRecord)
        .filter(models.ImageRecord.route_id == route_id)
        .options(selectinload(models.ImageRecord.detections))
        .order_by(models.ImageRecord.created_at.desc())
    )
    result = await db.execute(stmt)
    return result.scalars().all()


# Оригинальный CRUD для Detection
async def create_detection(
        db: AsyncSession,
        image_id: int,
        defect_type: str,
        yolo_confidence: float,
        classif_confidence: float,
        roi_path: str,
        gps_latitude: Optional[float] = None,
        gps_longitude: Optional[float] = None
):
    detection = models.Detection(
        image_id=image_id,
        defect_type=defect_type,
        yolo_confidence=yolo_confidence,
        classif_confidence=classif_confidence,
        roi_path=roi_path,
        gps_latitude=gps_latitude,
        gps_longitude=gps_longitude
    )
    db.add(detection)
    await db.commit()
    await db.refresh(detection)
    return detection

async def update_image_criticality(
        db: AsyncSession,
        image_id: int,
        criticality: int
):
    result = await db.execute(
        select(models.ImageRecord).filter(models.ImageRecord.id == image_id)
    )
    image = result.scalars().first()

    if image:
        image.criticality = criticality
        await db.commit()
        await db.refresh(image)

    return image

async def get_image(db: AsyncSession, image_id: int):
    result = await db.execute(
        select(models.ImageRecord).filter(models.ImageRecord.id == image_id)
    )
    return result.scalars().first()

async def get_detections_by_image(db: AsyncSession, image_id: int):
    stmt = (
        select(models.Detection)
        .options(selectinload(models.Detection.repair_request))
        .filter(models.Detection.image_id == image_id)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_all_images(db: AsyncSession):
    stmt = select(models.ImageRecord).options(selectinload(models.ImageRecord.detections))
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_all_detections_with_gps(db: AsyncSession):
    stmt = select(models.Detection).filter(
        models.Detection.gps_latitude.isnot(None),
        models.Detection.gps_longitude.isnot(None)
    )
    result = await db.execute(stmt)
    return result.scalars().all()

async def get_detection(db: AsyncSession, detection_id: int):
    result = await db.execute(
        select(models.Detection).filter(models.Detection.id == detection_id)
    )
    return result.scalars().first()

async def get_repair_request_by_detection(db: AsyncSession, detection_id: int):
    result = await db.execute(
        select(models.RepairRequest).filter(models.RepairRequest.detection_id == detection_id)
    )
    return result.scalars().first()

async def create_repair_request(db: AsyncSession, detection_id: int):
    repair_request = models.RepairRequest(
        detection_id=detection_id,
        status="open",
    )
    db.add(repair_request)
    await db.commit()
    await db.refresh(repair_request)
    return repair_request

async def get_all_repair_requests(db: AsyncSession):
    result = await db.execute(select(models.RepairRequest))
    return result.scalars().all()

async def get_repair_request_by_id(db: AsyncSession, request_id: int):
    result = await db.execute(
        select(models.RepairRequest).filter(models.RepairRequest.id == request_id)
    )
    return result.scalars().first()

async def update_repair_request_status(db: AsyncSession, request_id: int, status: str):
    repair_request = await get_repair_request_by_id(db, request_id)
    if repair_request:
        repair_request.status = status
        await db.commit()
        await db.refresh(repair_request)
    return repair_request

async def get_detections_statistics(db, start_date, end_date):
    # исходный запрос
    stmt = (
        select(
            cast(models.Detection.created_at, Date).label('date'),
            func.count(models.Detection.id).label('count')
        )
        .filter(
            models.Detection.created_at >= start_date,
            models.Detection.created_at <= end_date
        )
        .group_by(cast(models.Detection.created_at, Date))
        .order_by(cast(models.Detection.created_at, Date))
    )
    result = await db.execute(stmt)
    stats = result.all()

    # создать словарь по датам
    stats_dict = {stat.date: stat.count for stat in stats}

    # сгенерировать полный диапазон дат
    date_list = []
    current_date = start_date.date()
    end_date_only = end_date.date()
    while current_date <= end_date_only:
        date_list.append(current_date)
        current_date += timedelta(days=1)

    # формируем результат с заполнением нулями
    full_stats = [
        {'date': d, 'count': stats_dict.get(d, 0)}
        for d in date_list
    ]

    return full_stats

async def get_general_statistics(db: AsyncSession):
    total_requests_result = await db.execute(
        select(func.count(models.RepairRequest.id))
    )
    total_requests = total_requests_result.scalar()

    total_defects_result = await db.execute(
        select(func.count(models.Detection.id))
    )
    total_defects = total_defects_result.scalar()

    open_requests_result = await db.execute(
        select(func.count(models.RepairRequest.id))
        .filter(models.RepairRequest.status == "open")
    )
    open_requests = open_requests_result.scalar()

    closed_requests_result = await db.execute(
        select(func.count(models.RepairRequest.id))
        .filter(models.RepairRequest.status == "closed")
    )
    closed_requests = closed_requests_result.scalar()

    completed_requests_result = await db.execute(
        select(func.count(models.RepairRequest.id))
        .filter(models.RepairRequest.status == "completed")
    )
    completed_requests = completed_requests_result.scalar()

    return {
        'total_repair_requests': total_requests or 0,
        'total_defects': total_defects or 0,
        'open_repair_requests': open_requests or 0,
        'closed_repair_requests': closed_requests or 0,
        'completed_repair_requests': completed_requests or 0
    }

async def get_detections_gps_by_route(db: AsyncSession, route_id: int):
    stmt = (
        select(models.Detection)
        .join(models.ImageRecord, models.Detection.image_id == models.ImageRecord.id)
        .filter(models.ImageRecord.route_id == route_id)
        .options(joinedload(models.Detection.image))
    )
    result = await db.execute(stmt)
    return result.scalars().all()