import cv2
import sys

# Get user supplied values - to pass in as command line arguments
imagePath = sys.argv[1]
cascPath = sys.argv[2]

# Create the haar cascade - Creating cascade and Initializing it with my face cascade, this loads the face cascade into memory so it is ready to use. The cascade is an XML file that contains the data to detect faces
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image and converting it to grayscale, many operations in OpenCV are done in grayscale
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale( # General Function that detects objects, since I am calling it on the face cascade that is what it will detect
    gray, # First option is the grayscale image
    scaleFactor=1.1, # The second option, compensates for faces that are farther away from the camera
    minNeighbors=5, # Third option, the detection algorithm uses a moving window to detect object. this defines how many objects are detected near the current one before it declares the face found
    minSize=(30, 30) # Fourth option, this gives the size of each window
)

# Returns a list of rectangles in which it believes it found a face, And I loop over what is found to see if it was correct
print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces, using built-in function rectangle()
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Display what is found
cv2.imshow("Faces found", image)
cv2.waitKey(0)