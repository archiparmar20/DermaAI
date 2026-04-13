import os
import json
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from PIL import Image
import io
import torch
from torchvision import models, transforms
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='frontend_ui/dist', static_url_path='')
CORS(app)

# Global model and class mapping
MODEL = None
CLASS_MAPPING = None
IMG_SIZE = (224, 224)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(num_classes):
    """Build ResNet50 model (matches train.py architecture)"""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace final layer with architecture matching train.py
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(num_ftrs, 512),
        torch.nn.BatchNorm1d(512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(512, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 128),
        torch.nn.BatchNorm1d(128),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, num_classes)
    )
    
    return model

def load_app_model():
    """Load the trained model"""
    global MODEL, CLASS_MAPPING
    
    # Check for model in current directory first (priority)
    model_path = None
    current_dir_model = Path('run20260413_104741_FINAL_91pct.pth')
    
    if current_dir_model.exists():
        model_path = current_dir_model
    else:
        # Fallback to models directory
        model_dir = Path('models')
        if model_dir.exists():
            # Look for new run-based final models first
            final_models = list(model_dir.glob('run*_FINAL_*.pth'))
            if not final_models:
                # Fallback to old naming scheme
                final_models = list(model_dir.glob('m_final_*.pth'))
            
            if final_models:
                # Get the one with highest accuracy (last number in filename)
                model_path = sorted(final_models, key=lambda x: int(x.stem.split('_')[-1].replace('pct', '')))[-1]
    
    mapping_path = 'models/class_mapping.json'
    
    if os.path.exists(mapping_path):
        try:
            with open(mapping_path, 'r') as f:
                CLASS_MAPPING = json.load(f)
                # Convert keys to integers
                CLASS_MAPPING = {int(k): v for k, v in CLASS_MAPPING.items()}
            print("✓ Class mapping loaded")
        except Exception as e:
            print(f"❌ Error loading class mapping: {e}")
            return False
    else:
        print(f"❌ Class mapping not found at {mapping_path}")
        return False
    
    if model_path and os.path.exists(model_path):
        try:
            num_classes = len(CLASS_MAPPING)
            MODEL = build_model(num_classes)
            MODEL.load_state_dict(torch.load(model_path, map_location=DEVICE))
            MODEL = MODEL.to(DEVICE)
            MODEL.eval()
            print(f"✓ Model loaded successfully from: {model_path.name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"❌ Model not found")
        print("   Expected: run20260413_104741_FINAL_91pct.pth or models/run*_FINAL_*.pth")
        print("   Please place the model file in the project directory or train a new model: python train.py")
        return False
    
    return True

def preprocess_image(img):
    """Preprocess image for prediction"""
    try:
        # Apply transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(DEVICE)
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing: {e}")

def get_training_progress():
    """Read training history"""
    history_file = 'training_history.json'
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_loaded = MODEL is not None
    return jsonify({
        'status': 'ok' if model_loaded else 'model_not_loaded',
        'model_ready': model_loaded,
        'classes': len(CLASS_MAPPING) if CLASS_MAPPING else 0
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict skin disease from image"""
    if MODEL is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get image from request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Preprocess
        img_tensor = preprocess_image(img)
        if img_tensor is None:
            return jsonify({'error': 'Image processing failed'}), 400
        
        # Predict
        with torch.no_grad():
            outputs = MODEL(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, class_idx = torch.max(probabilities, 1)
            confidence = confidence.item()
            class_idx = class_idx.item()
        
        class_name = CLASS_MAPPING.get(class_idx, 'Unknown')
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities[0], 3)
        top_predictions = [
            {'class': CLASS_MAPPING.get(int(idx.item()), 'Unknown'), 
             'confidence': float(prob.item())}
            for prob, idx in zip(top_probs, top_indices)
        ]
        
        return jsonify({
            'success': True,
            'prediction': class_name,
            'confidence': confidence,
            'confidence_percent': f"{confidence*100:.2f}%",
            'top_3': top_predictions,
            'all_classes': list(CLASS_MAPPING.values())
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-history', methods=['GET'])
def training_history():
    """Get training history"""
    history = get_training_progress()
    if history:
        return jsonify({
            'success': True,
            'history': history,
            'total_epochs': len(history),
            'final_accuracy': history[-1]['val_accuracy'] if history else 0
        })
    return jsonify({'success': False, 'message': 'No training history found'}), 404

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    history = get_training_progress()
    final_accuracy = history[-1]['val_accuracy'] if history else 0
    
    return jsonify({
        'classes': len(CLASS_MAPPING) if CLASS_MAPPING else 0,
        'class_names': list(CLASS_MAPPING.values()) if CLASS_MAPPING else [],
        'image_size': IMG_SIZE,
        'final_validation_accuracy': f"{final_accuracy*100:.2f}%",
        'training_epochs': len(history) if history else 0
    })

@app.route('/', methods=['GET'])
def serve_frontend():
    """Serve the frontend"""
    frontend_path = 'frontend_ui/dist/index.html'
    if os.path.exists(frontend_path):
        with open(frontend_path, 'r') as f:
            return f.read()
    else:
        return jsonify({'error': 'Frontend not built. Run: cd frontend_ui && pnpm install && pnpm build'}), 404

@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static files from frontend dist"""
    dist_path = Path('frontend_ui/dist')
    if dist_path.exists():
        file_path = dist_path / path
        if file_path.exists() and file_path.is_file():
            return send_from_directory(dist_path, path)
    # Fallback to index.html for SPA routing
    index_path = Path('frontend_ui/dist/index.html')
    if index_path.exists():
        with open(index_path, 'r') as f:
            return f.read()
    return jsonify({'error': 'Not found'}), 404

if __name__ == '__main__':
    print("=" * 60)
    print("SKIN DISEASE DETECTION API")
    print("=" * 60)
    
    # Load model
    if load_app_model():
        print("\n✓ Server starting on http://localhost:5000")
        print("  Open browser to http://localhost:5000")
        print("\nAvailable endpoints:")
        print("  GET  /                   - Frontend UI")
        print("  GET  /api/health         - Health check")
        print("  POST /api/predict        - Predict disease (upload image)")
        print("  GET  /api/model-info     - Model information")
        print("  GET  /api/training-history - Training progress")
        print("\n")
        app.run(debug=False, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Could not start server - model not available")
        print("   Please train the model first: python train.py")
