import cv2
import pytesseract
import torch
import numpy as np
import imutils

# Load PyTorch model
model = torch.load('license_plate_model.pth')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Iterate through frames in video
while True:
    # Capture frame from video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    frame = imutils.resize(frame, width=600)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    frame_thresh = cv2.adaptiveThreshold(frame_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in frame
    contours, _ = cv2.findContours(frame_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find potential license plate
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)
            if 2.5 < w/h < 4:
                plate_img = frame_gray[y:y+h, x:x+w]
                plate_img = cv2.resize(plate_img, (128, 64))
                plate_img = plate_img / 255.0
                plate_tensor = torch.from_numpy(np.expand_dims(plate_img, axis=0)).type(torch.FloatTensor)
                with torch.no_grad():
                    output = model(plate_tensor)
                    prediction = output.data.max(1, keepdim=True)[1]
                    plate_number = ''.join(str(e) for e in prediction.tolist()[0])

                # Crop frame based on prediction box
                x1 = int(max(x - w * 0.1, 0))
                y1 = int(max(y - h * 0.1, 0))
                x2 = int(min(x + w * 1.1, frame.shape[1]))
                y2 = int(min(y + h * 1.1, frame.shape[0]))
                frame_crop = frame[y1:y2, x1:x2]

                # Apply OCR to get license plate number
                config = '--psm 11 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                plate_number = pytesseract.image_to_string(frame_crop, config=config)

                # Draw prediction box and license plate number on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_number, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('License Plate Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
