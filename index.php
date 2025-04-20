<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rice Leaf Disease Scanner</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .upload-form {
            text-align: center;
            margin: 20px 0;
        }
        .input-section {
            margin: 20px 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        }
        .button {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px;
            min-width: 200px;
        }
        .button:hover {
            background-color: #2980b9;
        }
        .camera-button {
            background-color: #2ecc71;
        }
        .camera-button:hover {
            background-color: #27ae60;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .preview-container {
            width: 100%;
            max-width: 400px;
            margin: 20px auto;
            position: relative;
        }
        #preview, #camera-preview {
            width: 100%;
            max-width: 400px;
            height: 300px;
            object-fit: contain;
            display: none;
            border-radius: 8px;
            border: 2px solid #ddd;
        }
        .camera-container {
            display: block;
            position: relative;
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
        }
        #camera-preview {
            width: 100%;
            max-width: 400px;
            height: 300px;
            object-fit: cover;
            border-radius: 8px;
            border: 2px solid #ddd;
            background-color: #000;
        }
        .camera-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 10px;
        }
        .tab-buttons {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .tab-button {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 10px 20px;
            cursor: pointer;
        }
        .tab-button.active {
            background-color: #3498db;
            color: white;
            border-color: #3498db;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .camera-error {
            color: #721c24;
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rice Leaf Disease Scanner</h1>
        
        <div class="tab-buttons">
            <button class="tab-button active" onclick="showTab('upload')">Upload Image</button>
            <button class="tab-button" onclick="showTab('camera')">Use Camera</button>
        </div>

        <div id="upload-tab" class="tab-content active">
            <form class="upload-form" id="uploadForm">
                <div class="input-section">
                    <input type="file" name="image" id="image" accept="image/*" style="display: none;">
                    <button type="button" class="button" onclick="document.getElementById('image').click()">
                        Choose Image from Gallery
                    </button>
                </div>
                <div class="preview-container">
                    <img id="preview" alt="Preview">
                </div>
                <button type="submit" class="button">Scan for Disease</button>
            </form>
        </div>

        <div id="camera-tab" class="tab-content">
            <div class="camera-container">
                <div class="camera-error" id="cameraError"></div>
                <video id="camera-preview" autoplay playsinline muted></video>
                <div class="camera-controls">
                    <button class="button camera-button" id="startCamera">Start Camera</button>
                    <button class="button camera-button" id="captureImage" style="display: none;">Take Photo</button>
                    <button class="button camera-button" id="retakePhoto" style="display: none;">Retake</button>
                    <button class="button" id="scanCaptured" style="display: none;">Scan Photo</button>
                </div>
                <canvas id="canvas" style="display: none;"></canvas>
            </div>
        </div>

        <div class="loading" id="loading">
            Scanning image... Please wait...
        </div>
        <div class="result" id="result"></div>
    </div>

    <script>
        // Tab functionality
        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-button').forEach(button => button.classList.remove('active'));
            
            document.getElementById(tabName + '-tab').classList.add('active');
            event.target.classList.add('active');
            
            if (tabName === 'camera') {
                document.getElementById('startCamera').style.display = 'inline-block';
                document.getElementById('cameraError').style.display = 'none';
            } else {
                stopCamera();
            }
        }

        // Camera handling
        let stream = null;
        const video = document.getElementById('camera-preview');
        const canvas = document.getElementById('canvas');
        const startButton = document.getElementById('startCamera');
        const captureButton = document.getElementById('captureImage');
        const retakeButton = document.getElementById('retakePhoto');
        const scanButton = document.getElementById('scanCaptured');
        const cameraError = document.getElementById('cameraError');

        // Initialize camera when the page loads
        document.addEventListener('DOMContentLoaded', function() {
            checkCamera();
        });

        // Check if the device has a camera
        async function checkCamera() {
            try {
                // First, check if mediaDevices is supported
                if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                    throw new Error('Camera API is not supported in your browser');
                }

                const devices = await navigator.mediaDevices.enumerateDevices();
                const cameras = devices.filter(device => device.kind === 'videoinput');
                
                if (cameras.length === 0) {
                    throw new Error('No camera found on this device');
                }

                // Log available cameras for debugging
                console.log('Available cameras:', cameras);
                
            } catch (err) {
                console.error('Camera check error:', err);
                cameraError.textContent = err.message || 'Error checking camera availability';
                cameraError.style.display = 'block';
                startButton.disabled = true;
            }
        }

        startButton.addEventListener('click', async () => {
            try {
                cameraError.style.display = 'none';
                
                // First try to get the back/environment camera
                try {
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: {
                            facingMode: { ideal: 'environment' },
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        },
                        audio: false
                    });
                } catch (backCameraError) {
                    console.log('Back camera failed, trying front camera:', backCameraError);
                    
                    // If back camera fails, try any available camera
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: true,
                        audio: false
                    });
                }

                // Set up video stream
                video.srcObject = stream;
                video.style.display = 'block';
                
                // Wait for video to be ready
                await video.play();
                
                startButton.style.display = 'none';
                captureButton.style.display = 'inline-block';
                
            } catch (err) {
                console.error('Camera access error:', err);
                cameraError.textContent = 'Could not access camera. Please ensure you have granted camera permissions.';
                cameraError.style.display = 'block';
                
                // Try to get more specific error information
                if (err.name === 'NotAllowedError') {
                    cameraError.textContent = 'Camera access denied. Please allow camera access in your browser settings.';
                } else if (err.name === 'NotFoundError') {
                    cameraError.textContent = 'No camera found on your device.';
                } else if (err.name === 'NotReadableError') {
                    cameraError.textContent = 'Camera is already in use by another application.';
                }
            }
        });

        captureButton.addEventListener('click', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            // Stop camera and hide video
            video.style.display = 'none';
            captureButton.style.display = 'none';
            retakeButton.style.display = 'inline-block';
            scanButton.style.display = 'inline-block';
        });

        retakeButton.addEventListener('click', () => {
            video.style.display = 'block';
            retakeButton.style.display = 'none';
            scanButton.style.display = 'none';
            captureButton.style.display = 'inline-block';
        });

        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                video.srcObject = null;
                video.style.display = 'none';
                captureButton.style.display = 'none';
                retakeButton.style.display = 'none';
                scanButton.style.display = 'none';
                startButton.style.display = 'inline-block';
            }
        }

        scanButton.addEventListener('click', () => {
            canvas.toBlob((blob) => {
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');
                scanImage(formData);
            }, 'image/jpeg', 0.95);
        });

        // File upload handling
        document.getElementById('image').addEventListener('change', function(e) {
            const preview = document.getElementById('preview');
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }
                reader.readAsDataURL(file);
            }
        });

        // Form submission handling
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            scanImage(formData);
        });

        function scanImage(formData) {
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.style.display = 'none';

            fetch('scan.php', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                result.style.display = 'block';
                if (data.status === 'success') {
                    result.className = 'result success';
                    let resultHtml = `
                        <h3>Scan Results:</h3>
                        <p><strong>Disease:</strong> ${data.disease}</p>
                        <p><strong>Confidence:</strong> ${data.confidence}</p>
                    `;
                    if (data.all_probabilities) {
                        resultHtml += '<h4>All Probabilities:</h4><ul>';
                        for (const [disease, prob] of Object.entries(data.all_probabilities)) {
                            resultHtml += `<li>${disease}: ${prob}</li>`;
                        }
                        resultHtml += '</ul>';
                    }
                    result.innerHTML = resultHtml;
                } else {
                    result.className = 'result error';
                    result.innerHTML = `<p>Error: ${data.error}</p>`;
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                result.style.display = 'block';
                result.className = 'result error';
                result.innerHTML = `<p>Error: ${error.message}</p>`;
            });
        }
    </script>
</body>
</html> 
</html> 