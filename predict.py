import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
from config import MODEL_PATH, CLASS_TO_IDX, NUM_CLASSES
from utils import TEST_TRANSFORMS

def load_model(device):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = TEST_TRANSFORMS(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    
    pred_class = list(CLASS_TO_IDX.keys())[list(CLASS_TO_IDX.values()).index(pred_idx.item())]
    return pred_class, confidence.item()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    
    pred, conf = predict_image(image_path, model, device)
    print(f"Prediction: {pred} (confidence: {conf:.2%})")
