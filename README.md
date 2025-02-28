# Arabic OCR System with KHATT Dataset

This project implements an Arabic OCR (Optical Character Recognition) system that works with the KHATT dataset. It allows users to extract text from Arabic images using multiple OCR models.

## Features

- Support for multiple OCR models:
  - TrOCR: A transformer-based OCR model
  - AR-OCR: A specialized model for Arabic text recognition
  - Tesseract: An open-source OCR engine with Arabic language support
- User-friendly interface for uploading and processing images
- Real-time results display with proper RTL text rendering
- Responsive design that works on desktop and mobile devices

## Technology Stack

- **Frontend**: React with TypeScript, Tailwind CSS
- **Backend**: Flask API
- **OCR Models**: TrOCR, AR-OCR, Tesseract

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- Python (v3.8 or higher)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/arabic-ocr-system.git
   cd arabic-ocr-system
   ```

2. Install frontend dependencies:
   ```
   npm install
   ```

3. Install backend dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

### Running the Application

1. Start the backend server:
   ```
   npm run start-api
   ```

2. In a new terminal, start the frontend development server:
   ```
   npm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`

## Project Structure

```
arabic-ocr-system/
├── backend/                # Flask backend
│   ├── app.py              # Main Flask application
│   └── requirements.txt    # Python dependencies
├── public/                 # Public assets
├── src/                    # React frontend
│   ├── components/         # Reusable components
│   ├── pages/              # Page components
│   ├── App.tsx             # Main application component
│   └── main.tsx            # Entry point
└── package.json            # Node.js dependencies
```

## Acknowledgements

- KHATT Dataset for providing the training data
- The developers of TrOCR, AR-OCR, and Tesseract
