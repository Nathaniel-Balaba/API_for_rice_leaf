# Rice Leaf Disease Classification API

This API provides a service for classifying diseases in rice leaves using deep learning. It can detect three types of diseases:
- Bacterial leaf blight
- Brown spot
- Leaf smut

## Features
- Image-based disease classification
- Web interface for easy image upload
- Camera support for direct capture
- Mobile-friendly design

## Project Structure
```
rice_leaf_diseases/
├── api.py              # FastAPI backend
├── index.php           # Web interface
├── scan.php            # PHP endpoint for image processing
├── requirements.txt    # Python dependencies
└── rice_leaf_model.pth # Trained model (not included in repo)
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rice-leaf-diseases.git
cd rice-leaf-diseases
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained model file (`rice_leaf_model.pth`) in the project directory

4. Run the API:
```bash
python api.py
```

5. Access the web interface at `http://localhost:8000`

## API Usage

The API provides the following endpoints:

- `GET /`: Welcome message and API information
- `POST /predict`: Upload an image for disease classification

## Web Interface

The web interface (`index.php`) provides:
- Image upload from gallery
- Camera capture support
- Real-time disease classification
- Confidence scores and probabilities

## Requirements

- Python 3.7+
- PHP 7.4+
- Modern web browser with camera support
- FastAPI and other Python packages (see requirements.txt)

## License

This project is licensed under the MIT License - see the LICENSE file for details.