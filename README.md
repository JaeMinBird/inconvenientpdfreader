# PDF Book Reader with Gesture Control

A Python application that converts PDFs into a traditional book reading experience with webcam-based hand gesture controls for page turning.

## Features

- **Traditional Book Layout**: Displays PDF pages in a two-page spread like a real book
- **Gesture Controls**: Wave your hand left or right to turn pages, and put your finger to your lips to you lick your finger before turning a page (required each page turn)
- **Real-time Webcam Integration**: See your hand movements with gesture detection overlay
- **High-Quality PDF Rendering**: Supports both PyMuPDF (recommended) and PyPDF2 for PDF processing

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. For better PDF image quality, install PyMuPDF (optional but recommended):
```bash
pip install PyMuPDF
```

## Usage

Run the application with a PDF file:
```bash
python main.py path/to/your/document.pdf
```

### Controls

- **Hand Gestures**:
  - Wave hand **RIGHT** → Next page
  - Wave hand **LEFT** → Previous page
- **Keyboard**:
  - **SPACE** → Toggle webcam view on/off
  - **ESC** → Exit application

### Command Line Options

```bash
python main.py document.pdf --width 1400 --height 900
```

- `--width`: Set window width (default: 1200)
- `--height`: Set window height (default: 800)

## How It Works

1. **PDF Processing**: Extracts pages from PDF and converts them to images
2. **Gesture Detection**: Uses MediaPipe to detect hand landmarks and track movement
3. **Book Viewer**: Renders pages in a traditional book layout using Pygame
4. **Real-time Integration**: Combines webcam feed with gesture recognition for page control

## Requirements

- Python 3.7+
- Webcam
- Dependencies listed in `requirements.txt`

## Troubleshooting

- **Webcam not working**: Ensure your webcam is not being used by another application
- **PDF not loading**: Make sure the PDF file path is correct and the file is not corrupted
- **Poor gesture detection**: Ensure good lighting and keep your hand clearly visible to the camera
- **Low image quality**: Install PyMuPDF for better PDF rendering quality

## File Structure

```
pdf-book-reader/
├── main.py              # Main application entry point
├── pdf_reader.py        # PDF processing and page management
├── gesture_detector.py  # Hand gesture detection using MediaPipe
├── book_viewer.py       # Book layout and rendering using Pygame
├── requirements.txt     # Python dependencies
└── README.md           # This file
```
