import cv2
import sys

# Creating the face cascade used to detect faces
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Setting OpenCv to video source - default webcam for easy use and testing
video_capture = cv2.VideoCapture(0)

while True:
    # Using the read() function to capture and read the video on each loop one frame at a time and produce
    # This can have a return, but since I am reading from the webcam it does not matter
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( # General Function that detects objects, since I am calling it on the face cascade that is what it will detect
        gray, # First option is the grayscale image
        scaleFactor=1.1, # The second option, compensates for faces that are farther away from the camera
        minNeighbors=5, # Third option, the detection algorithm uses a moving window to detect object. this defines how many objects are detected near the current one before it declares the face found
        minSize=(30, 30) # Fourth option, this gives the size of each window
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

# If q key is pressed, I want to exit the script
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()