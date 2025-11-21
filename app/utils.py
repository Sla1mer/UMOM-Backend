import os
import uuid
from typing import Optional, Tuple
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

import cv2
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

allowed_main_classes = {
    0: "vibration_damper",
    1: "festoon_insulators",
    2: "traverse",
    3: "nest",
    4: "safety_sign",
    7: "polymer_insulators"
}

CLASSIF_NAMES = ['chip', 'corrosion', 'crack', 'undefined']
device = torch.device("mps" if torch.has_mps else "cpu")

def predict_defect(classif_model, img_np):
    img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = classif_model(img_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(prob, 1)
    return CLASSIF_NAMES[predicted.item()], float(confidence)

def save_roi(roi_img):
    os.makedirs("rois", exist_ok=True)
    roi_filename = f"{uuid.uuid4()}.jpg"
    roi_path = os.path.join("rois", roi_filename)
    cv2.imwrite(roi_path, roi_img)
    return roi_path


def get_gps_from_bytes(image_bytes: bytes) -> Optional[Tuple[float, float]]:
    """
    Извлекает GPS координаты из байтов изображения

    Returns:
        Tuple[float, float]: (latitude, longitude) или None если GPS нет
    """
    try:
        from io import BytesIO
        image = Image.open(BytesIO(image_bytes))
        exif_data = image._getexif()

        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            tag_name = TAGS.get(tag, tag)
            if tag_name == 'GPSInfo':
                for gps_tag in value:
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    gps_info[gps_tag_name] = value[gps_tag]

        if not gps_info:
            return None

        lat = _convert_to_degrees(gps_info.get('GPSLatitude'))
        lon = _convert_to_degrees(gps_info.get('GPSLongitude'))

        if lat is None or lon is None:
            return None

        if gps_info.get('GPSLatitudeRef') == 'S':
            lat = -lat
        if gps_info.get('GPSLongitudeRef') == 'W':
            lon = -lon

        return (lat, lon)

    except Exception as e:
        print(f"Ошибка извлечения GPS из bytes: {e}")
        return None


def _convert_to_degrees(value):
    """
    Конвертирует GPS координаты из формата EXIF в десятичные градусы
    """
    if not value:
        return None

    try:
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)
    except:
        return None
