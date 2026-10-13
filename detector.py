import cv2
from qreader import QReader

# Initialize the QReader
qreader = QReader()

# Load the image
image = cv2.cvtColor(cv2.imread("my_qr_code.png"), cv2.COLOR_BGR2RGB)

# Decode the QR code and get the result and bounding box
decoded_text, detection_results = qreader.detect_and_decode(image=image, return_detections=True)

if decoded_text:
    for text, detection in zip(decoded_text, detection_results):
        print(f"Detected QR Code data: {text}")

        # Draw a bounding box around the detected QR code
        if detection is not None:
            # The detection result is a dictionary with a 'bbox_xyxy' key
            bbox = detection['bbox_xyxy']
            x_min, y_min, x_max, y_max = [int(p) for p in bbox]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the image with the bounding box
    cv2.imshow("QReader Detector", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No QR code detected.")