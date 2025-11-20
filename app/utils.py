import os
import uuid
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
