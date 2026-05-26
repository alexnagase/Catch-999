import cv2
import easyocr
import re
import os
import time
from ultralytics import YOLO

# Create a local directory to save the matches
SAVE_DIR = "Captured_Plates"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    # 1. Get the starting number from the user
    start_input = input("Enter the starting 3-digit number (e.g., 003): ")
    try:
        current_target = int(start_input)
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return

    # 2. Initialize Models
    print("Loading models... (This may take a moment on first run)")
    # Using the nano version of YOLOv8 for maximum processing speed on a laptop
    yolo_model = YOLO("yolov8n.pt") 
    # Initialize EasyOCR reader for English text
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if PyTorch MPS is configured

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"\n--- YOLOv8 + EasyOCR Pipeline Active ---")
    print(f"Looking for: {current_target:03d}")
    print("Press 'q' in the video window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        target_str = f"{current_target:03d}"

        # 4. YOLO Object Detection Step (Localization)
        # Class 57 in the default COCO dataset is 'car', but we can check for plates or vehicles.
        # For a specialized license plate tracker, you can substitute a custom license plate weights file (.pt)
        results = yolo_model(frame, stream=True, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Extract coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Check if detected object meets a confidence threshold (e.g., > 40%)
                # For this baseline script, we treat detected regions of interest as processing zones
                if conf > 0.40:
                    # Crop the detected object out of the main frame
                    cropped_zone = frame[y1:y2, x1:x2]
                    
                    if cropped_zone.size == 0:
                        continue

                    # 5. EasyOCR Step (Recognition)
                    # Pass the isolated cropped zone instead of the massive full frame
                    ocr_results = reader.readtext(cropped_zone)
                    
                    # Consolidate text strings from the crop area
                    extracted_text = " ".join([res[1] for res in ocr_results]).upper()
                    
                    # 6. Sequential Match Evaluation
                    if re.search(rf"{target_str}", extracted_text):
                        print(f"MATCH FOUND: {target_str} inside raw text: '{extracted_text}'")
                        
                        # Highlight the match on the master frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        
                        # Save the full capture context
                        filename = os.path.join(SAVE_DIR, f"Plate_{target_str}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"Saved image to: {filename}")
                        
                        # Update the operational target state
                        current_target += 1
                        print(f"--> Upgraded target to: {current_target:03d}\n")
                        time.sleep(2)
                        break

        # Display operational status on the screen preview
        cv2.putText(frame, f"Hunting: {target_str}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLOv8 + EasyOCR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()