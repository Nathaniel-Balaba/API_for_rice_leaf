from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import uvicorn
import os
from torchvision.models import resnet18, ResNet18_Weights

# Define the CNN model (same as in train.py)
class RiceLeafCNN(nn.Module):
    def __init__(self, num_classes):
        super(RiceLeafCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

app = FastAPI(title="Rice Leaf Disease Classification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the disease classification model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
disease_model = RiceLeafCNN(num_classes=3).to(device)
model_path = os.path.join(os.path.dirname(__file__), 'rice_leaf_model.pth')
disease_model.load_state_dict(torch.load(model_path, map_location=device))
disease_model.eval()

# Load pre-trained ResNet model for general image classification
validation_model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
validation_model.eval()

# Define the image transforms
disease_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

# Define thresholds
CONFIDENCE_THRESHOLD = 90.0  # Increased threshold for disease classification
MAX_ENTROPY_THRESHOLD = 0.6  # Reduced threshold for entropy (more strict)
COLOR_THRESHOLD = 0.3  # Threshold for green color validation

def is_likely_plant(image):
    # Convert image to RGB array
    img_array = np.array(image)
    
    # Calculate the average green channel value relative to other channels
    r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
    green_ratio = np.mean(g) / (np.mean(r) + np.mean(b) + 1e-6)
    
    # Check if image has significant green content
    return green_ratio > COLOR_THRESHOLD

def validate_with_resnet(image, validation_model, device):
    # Transform image for ResNet
    img_tensor = validation_transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = validation_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predicted class
        _, predicted = torch.max(outputs, 1)
        
        # Check if any of the top 5 predictions are plant-related
        _, top5_idx = torch.topk(probabilities, 5)
        plant_related = any(idx.item() in range(970, 990) for idx in top5_idx[0])  # ImageNet plant classes
        
        return plant_related

@app.get("/")
async def root():
    return {
        "message": "Welcome to Rice Leaf Disease Classification API",
        "status": "active",
        "endpoints": {
            "predict": "/predict (POST) - Upload an image to classify rice leaf disease"
        }
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Validate if image likely contains a plant
        if not is_likely_plant(image):
            return {
                "disease": "Unknown (Not a plant image)",
                "confidence": "0.00%",
                "status": "success"
            }
            
        # Validate with ResNet if image contains plant-like features
        if not validate_with_resnet(image, validation_model, device):
            return {
                "disease": "Unknown (Not a rice leaf image)",
                "confidence": "0.00%",
                "status": "success"
            }
        
        # Process image for disease classification
        image_tensor = disease_transform(image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = disease_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Calculate entropy of the prediction
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)
            normalized_entropy = entropy / np.log(len(class_names))
            
            # Get the highest confidence and its index
            confidence, predicted = torch.max(probabilities, 1)
            confidence_value = confidence.item() * 100
            
            # Check if prediction is uncertain (high entropy) or low confidence
            if normalized_entropy > MAX_ENTROPY_THRESHOLD or confidence_value < CONFIDENCE_THRESHOLD:
                prediction = "Unknown (Not a rice leaf or unclear image)"
                confidence_value = 0.0
            else:
                prediction = class_names[predicted.item()]

        return {
            "disease": prediction,
            "confidence": f"{confidence_value:.2f}%",
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 