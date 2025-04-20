<?php
header('Content-Type: application/json');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['status' => 'error', 'error' => 'Only POST requests are allowed']);
    exit;
}

if (!isset($_FILES['image'])) {
    echo json_encode(['status' => 'error', 'error' => 'No image file uploaded']);
    exit;
}

$file = $_FILES['image'];

if ($file['error'] !== UPLOAD_ERR_OK) {
    echo json_encode(['status' => 'error', 'error' => 'File upload failed']);
    exit;
}

// Create CURLFile object
$cfile = new CURLFile($file['tmp_name'], $file['type'], $file['name']);

// Initialize cURL
$ch = curl_init();

// Set cURL options
curl_setopt($ch, CURLOPT_URL, 'https://your-app-name.onrender.com/predict');  // Replace with your Render.com URL
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_POSTFIELDS, ['image' => $cfile]);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);

// Execute cURL request
$response = curl_exec($ch);

// Check for cURL errors
if (curl_errno($ch)) {
    echo json_encode(['status' => 'error', 'error' => 'API request failed: ' . curl_error($ch)]);
    curl_close($ch);
    exit;
}

// Close cURL
curl_close($ch);

// Forward the API response
echo $response;
?> 