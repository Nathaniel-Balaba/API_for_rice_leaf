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
        .file-input {
            margin: 20px 0;
        }
        .submit-btn {
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .submit-btn:hover {
            background-color: #2980b9;
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
        .preview-image {
            max-width: 300px;
            max-height: 300px;
            margin: 20px auto;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Rice Leaf Disease Scanner</h1>
        <form class="upload-form" action="scan.php" method="POST" enctype="multipart/form-data" id="uploadForm">
            <div class="file-input">
                <input type="file" name="image" id="image" accept="image/*" required>
            </div>
            <img id="preview" class="preview-image">
            <button type="submit" class="submit-btn">Scan for Disease</button>
        </form>
        <div class="loading" id="loading">
            Scanning image... Please wait...
        </div>
        <div class="result" id="result"></div>
    </div>

    <script>
        // Preview image before upload
        document.getElementById('image').addEventListener('change', function(e) {
            const preview = document.getElementById('preview');
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                }
                reader.readAsDataURL(file);
            }
        });

        // Handle form submission
        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
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
                    result.innerHTML = `
                        <h3>Scan Results:</h3>
                        <p><strong>Disease:</strong> ${data.disease}</p>
                        <p><strong>Confidence:</strong> ${data.confidence}</p>
                    `;
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
        });
    </script>
</body>
</html> 