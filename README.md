# Rice Leaf Disease Classification API

This API provides a service for classifying diseases in rice leaves using deep learning. It can detect three types of diseases:
- Bacterial leaf blight
- Brown spot
- Leaf smut

## Features
- Image-based disease classification
- Multiple validation layers to ensure accurate predictions
- Handles non-rice leaf images by returning "Unknown"
- High confidence threshold for reliable predictions

## Requirements
```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
tqdm>=4.65.0
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Nathaniel-Balaba/rice_leaf-API.git
cd rice_leaf-API
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the API server:
```bash
python api.py
```

2. The API will be available at `http://localhost:8000`

3. Use the `/predict` endpoint to classify rice leaf images:
- Method: POST
- Endpoint: `/predict`
- Input: Form data with 'file' field containing the image

## API Response
The API returns JSON responses in the following format:
```json
{
    "disease": "Disease name or Unknown",
    "confidence": "Confidence percentage",
    "status": "success or error"
}
```

## Model Details
- Uses a custom CNN architecture for disease classification
- Implements ResNet18 for plant feature validation
- Multiple validation layers for accurate classification
- Confidence threshold: 90%
- Handles non-plant and non-rice leaf images 