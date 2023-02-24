import cv2
import datetime
import pytesseract
import pymysql

# Connect to the MySQL database
conn = pymysql.connect(
    host='localhost',
    user='silas',
    password='Silas08072002',
    database='platedb'
)

# Initialize the camera
camera = cv2.VideoCapture(0)

# Read the first frame
ret, frame = camera.read()

# Convert the first frame to grayscale
prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    # Read a new frame from the camera
    ret, frame = camera.read()
    
    # Convert the new frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate the difference between the previous and current frames
    frame_delta = cv2.absdiff(prev_gray, gray)
    
    # Threshold the difference to detect moving vehicles
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours in the thresholded difference image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Check if the contour is large enough to be a moving vehicle
        #if cv2.contourArea(c) < 500:
        #    continue
        
        # Draw a bounding box around the contour
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Extract the license plate from the bounding box
        license_plate = gray[y:y+h, x:x+w]
        
        # Use PyTesseract to extract the text from the license plate
        text = pytesseract.image_to_string(license_plate)
        
        # Check if the extracted text is a license plate number
        if len(text) == 7:
            print("License plate:", text)
            
            # Get the current date and time
            now = datetime.datetime.now()
            
            # Check if the license plate is already in the database
            with conn.cursor() as cursor:
                sql = "SELECT id, count FROM license_plates WHERE plate_number=%s"
                cursor.execute(sql, (text))
                result = cursor.fetchone()
                
                if result:
                    # Update the count of the license plate in the database
                    sql = "UPDATE license_plates SET count=%s, capture_time=%s WHERE id=%s"
                    cursor.execute(sql, (result[1]+1, now, result[0]))
