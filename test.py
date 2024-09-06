from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Open the webcam (usually index 0 for the default camera)
cap = cv2.VideoCapture(0)

# Load YOLO model for detecting objects related to personal protective equipment (PPE)
model = YOLO("cons_safety.pt")

# Class names for different objects detected by the model
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']

# Default color for drawing bounding boxes
myColor = (0, 0, 255)

# Create folder to save images if not already present
output_folder = "safety_compliance"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Main loop to process each frame from the webcam
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    if not success:
        break  # If there's an issue with the webcam, exit the loop

    # Resize the frame to make detection faster (you can adjust the scale factor for speed vs accuracy)
    img_resized = cv2.resize(img, (640, 480))

    # Perform object detection using YOLO on the resized frame
    results = model(img_resized, stream=True)

    # Process the results of object detection
    for r in results:
        # Extract bounding box information for each detected object
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Calculate confidence and class index
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Set color based on the class of the detected object
            if conf > 0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0, 255)  # Red for non-compliance
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == "Mask":
                    myColor = (0, 255, 0)  # Green for compliance

                    # Save the image if safety compliance is detected
                    image_path = os.path.join(output_folder, f"{currentClass}_{conf}.jpg")
                    cv2.imwrite(image_path, img)  # Save the original frame

                else:
                    myColor = (255, 0, 0)  # Blue for other objects

                # Display the class name and confidence on the image
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1, colorB=myColor,
                                   colorT=(255, 255, 255), colorR=myColor, offset=5)

                # Draw bounding box around the detected object
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3)

    # Display the annotated image
    cv2.imshow("Image", img)

    # Wait for a key press and continue the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to quit

# Release the webcam and close any open windows
cap.release()
cv2.destroyAllWindows()
