import cv2

# We use cv2.CAP_V4L2 to bypass the GStreamer memory bugs
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Lower the resolution to save memory
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not connect to the camera.")
else:
    print("Success! Showing feed. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Plane Tracker Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()