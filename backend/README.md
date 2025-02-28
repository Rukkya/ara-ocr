# Arabic OCR Backend

This is the backend component of the Arabic OCR system. It provides a Flask API that integrates multiple OCR models for Arabic text recognition.

## Features

- Integration with multiple OCR models:
  - TrOCR: A transformer-based OCR model from Microsoft
  - AR-OCR: A specialized model for Arabic text recognition
  - Tesseract: An open-source OCR engine with Arabic language support
- RESTful API for image processing
- Error handling and logging
- Health check endpoint

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR with Arabic language support

### Installing Tesseract

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-ara  # Arabic language data
```

#### On macOS:
```bash
brew install tesseract
brew install tesseract-lang  # Includes Arabic
```

#### On Windows:
1. Download and install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
2. Make sure to select Arabic language during installation
3. Add Tesseract to your PATH environment variable

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python app.py
```

The server will start on http://localhost:5000

## API Endpoints

### POST /ocr
Process an image with OCR.

**Request:**
- Form data:
  - `image`: The image file to process
  - `model`: The OCR model to use (trocr, ar-ocr, or tesseract)

**Response:**
```json
{
  "text": "Extracted Arabic text",
  "model": "trocr"
}
```

### GET /health
Check if the server is running.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /models
List available OCR models.

**Response:**
```json
{
  "trocr": {
    "name": "TrOCR",
    "description": "Transformer-based OCR model for Arabic text",
    "status": "available"
  },
  "ar-ocr": {
    "name": "AR-OCR",
    "description": "Specialized model for Arabic text recognition",
    "status": "available"
  },
  "tesseract": {
    "name": "Tesseract",
    "description": "Open-source OCR engine with Arabic language support",
    "status": "available"
  }
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request (missing image, invalid model, etc.)
- 500: Internal Server Error (processing error, model loading error, etc.)

## Fallback Mechanism

If a model fails to load or process an image, the system will fall back to mock implementations to ensure the API remains functional.