"""
    @Author:        Silas Rodriguez
    @Description:   Use OpenCV to identify playing cards
"""
# Import the libraries to use
import cv2
import os

# Define the path to the folder containing card images
image_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Load the binarized card images for faces and values
face_images = {
    'diamond': cv2.imread(os.path.join(image_folder, 'Diamonds.jpg'), 0),
    'clubs': cv2.imread(os.path.join(image_folder, 'Clubs.jpg'), 0),
    'hearts': cv2.imread(os.path.join(image_folder, 'Hearts.jpg'), 0),
    'spades': cv2.imread(os.path.join(image_folder, 'Spades.jpg'), 0),
}
value_images = {
    '2': cv2.imread(os.path.join(image_folder, '2.jpg'), 0),
    '3': cv2.imread(os.path.join(image_folder, '3.jpg'), 0),
    '4': cv2.imread(os.path.join(image_folder, '4.jpg'), 0),
    '5': cv2.imread(os.path.join(image_folder, '5.jpg'), 0),
    '6': cv2.imread(os.path.join(image_folder, '6.jpg'), 0),
    '7': cv2.imread(os.path.join(image_folder, '7.jpg'), 0),
    '8': cv2.imread(os.path.join(image_folder, '8.jpg'), 0),
    '9': cv2.imread(os.path.join(image_folder, '9.jpg'), 0),
    '10': cv2.imread(os.path.join(image_folder, '10.jpg'), 0),
    'jack': cv2.imread(os.path.join(image_folder, 'jack.jpg'), 0),
    'queen': cv2.imread(os.path.join(image_folder, 'queen.jpg'), 0),
    'king': cv2.imread(os.path.join(image_folder, 'king.jpg'), 0),
    'ace': cv2.imread(os.path.join(image_folder, 'ace.jpg'), 0),
}

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a separate window for the thresholded image
cv2.namedWindow('Thresholded Image', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Apply bilateral filtering to the frame
    frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to the grayscale frame
    thresh_frame = cv2.adaptiveThreshold(gray_frame, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Expand the black background to cover the whole top of the frame
    frame[0:60, :] = (0, 0, 0)

    # Loop through the face and value images to find matches
    best_match_face = None
    best_match_value = None
    best_match_face_score = float('-inf')  # Initialize to negative infinity
    best_match_value_score = float('-inf')  # Initialize to negative infinity

    for face_name, face_image in face_images.items():
        # Try to match the face in the entire frame
        face_match = cv2.matchTemplate(thresh_frame, face_image, cv2.TM_CCOEFF_NORMED)
        
        # Find the maximum similarity score
        _, max_val_face, _, _ = cv2.minMaxLoc(face_match)

        # Update the best match for faces if a better match is found
        if max_val_face > best_match_face_score:
            best_match_face_score = max_val_face
            best_match_face = face_name

    for value_name, value_image in value_images.items():
        # Try to match the value in the entire frame
        value_match = cv2.matchTemplate(thresh_frame, value_image, cv2.TM_CCOEFF_NORMED)
        
        # Find the maximum similarity score
        _, max_val_value, _, _ = cv2.minMaxLoc(value_match)

        # Update the best match for values if a better match is found
        if max_val_value > best_match_value_score:
            best_match_value_score = max_val_value
            best_match_value = value_name

    # Display the thresholded image in the 'Thresholded Image' window
    cv2.imshow('Thresholded Image', thresh_frame)

    # Display the result with improved readability
    cv2.putText(frame, f"Face: {best_match_face}, Value: {best_match_value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Card Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()