# Catch-999: Sequential License Plate Scanner

A lightweight, local Computer Vision pipeline designed to hunt for sequentially numbered license plates in real-time. 

This script uses a live webcam feed to continuously scan for a target 3-digit number (e.g., `001`, `002`, `003`). It leverages a two-stage machine learning process: **YOLOv8** for rapid object localization (finding the plate) and **EasyOCR** for accurate text extraction. When a match is found, the frame is saved locally, and the target automatically increments.

## Features
* **100% Local Processing:** No API keys, no monthly limits, and no internet connection required after the initial model download.
* **Optimized Two-Stage Pipeline:** Uses YOLOv8 to tightly crop passing vehicles/plates before running OCR, significantly reducing CPU load and improving read accuracy.
* **Sequential State Management:** Automatically increments your target number upon a successful read.
* **USB Webcam Support:** Easily configurable to use external webcams or capture cards.

## Prerequisites
* Python 3.8+
* macOS, Linux, or Windows (Apple Silicon/M-Series highly recommended for native PyTorch MPS acceleration).

## Installation

1. Clone or download this repository.
2. Install the required machine learning and computer vision dependencies:

    ```bash
    pip install ultralytics easyocr opencv-python
    ```

    *Note: On the very first run, the script will automatically download the lightweight YOLOv8 nano model (`yolov8n.pt`) and the EasyOCR English language pack.*

## Usage

Run the scanner directly from your terminal:

```bash
python3 plate_scanner.py
