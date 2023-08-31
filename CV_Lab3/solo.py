# Main file imports
import cv2
import easyocr as ocr

def main():
    #initialize cv and ocr reader objects
    cap = cv2.VideoCapture(0)  # Use the appropriate camera index if not the default.

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # Perform card detection using template matching.
        # You need to have a template image of the playing card design.
        template = cv2.imread('card_template.jpg', cv2.IMREAD_GRAYSCALE)
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc

        card_width, card_height = template.shape[::-1]  # Template shape is (width, height).

        # Extract the card region from the frame.
        card_region = frame[y:y+card_height, x:x+card_width]

        # Perform OCR on the card region using EasyOCR.
        reader = ocr.Reader(lang_list=['en'])  # Choose appropriate languages.
        result = reader.readtext(card_region)

        # Process OCR results to identify card value and face.

        # Overlay the card value and face on the frame.
        card_value = "5"  # Example card value.
        card_face = "Hearts"  # Example card face.
        cv2.putText(frame, f"{card_value} of {card_face}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(frame, (x, y), (x + card_width, y + card_height), (0, 255, 0), 2)

        # Display the frame with overlays.
        cv2.imshow('Card Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()