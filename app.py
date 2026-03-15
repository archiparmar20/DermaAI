from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import gradio as gr
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import numpy as np
import io
from config import MODEL_PATH, NUM_CLASSES, CLASS_TO_IDX
from utils import TEST_TRANSFORMS, predict_image

def load_model(device):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

app = FastAPI(title="SkinAI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    if model is None:
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    load_model()
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        pred_idx, conf = predict_image(model, image, device)
        pred_class = list(CLASS_TO_IDX.keys())[pred_idx]
        # Top 3 probs
        image_tensor = TEST_TRANSFORMS(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top3_probs, top3_idx = torch.topk(probs, 3)
        results = {list(CLASS_TO_IDX.keys())[int(idx)]: f"{prob:.1%}" for idx, prob in zip(top3_idx[0], top3_probs[0])}
        return JSONResponse(content={"top_predictions": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gradio
def predict_gradio(img):
    load_model()
    pred_idx, conf = predict_image(model, img, device)
    pred_class = list(CLASS_TO_IDX.keys())[pred_idx]
    return {pred_class: f"{conf:.1%}"}

demo = gr.Blocks(title="Derma AI")
with demo:
    gr.Markdown("# 🩹 Derma AI")
    # ... (keep original Gradio UI)

if __name__ == "__main__":
    # Run both
    gr.IntegrationPanel(app, elem_id="fastapi_app")
    uvicorn.run(app, host="0.0.0.0", port=8000)
