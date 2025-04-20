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

# Define the CNN model
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

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class names
class_names = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']

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
        
        # Transform the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = disease_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            prediction = class_names[predicted.item()]
            confidence_value = confidence.item() * 100
            
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