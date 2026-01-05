from deepface import DeepFace
import cv2

# Initialize the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Press 'q' to quit...")

while True:
    # 1. Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # 2. Analyze the frame with DeepFace
    try:
        # actions=['emotion'] tells it to only look for feelings
        # enforce_detection=False prevents it from crashing if it doesn't see a face for a second
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        
        # Get the dominant emotion
        emotion = result[0]['dominant_emotion']
        
        # 3. Draw the text on the video
        cv2.putText(frame, 
                    f"Emotion: {emotion}", 
                    (50, 50),  # Coordinates
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,         # Font scale
                    (0, 255, 0), # Color (Green)
                    2)         # Thickness
    except:
        pass # If something goes wrong (like no face), just keep going

    # 4. Show the video
    cv2.imshow('My AI Emotion Detector', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()