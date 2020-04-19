"""Using dlib shape predictor to predict facial landmarks in images."""
# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/sample.jpg

import cv2
import argparse
import dlib
import imutils
import numpy as np 
from imutils import face_utils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to the image")
args = vars(ap.parse_args())

# init dlib's HOG face detector and the facial landmarks predictor
face_hog_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = face_hog_detector(gray, 1)

for (i, rect) in enumerate(rects):
    # get dlib's shape object contains facial landmarks coordinates
    shape = landmark_predictor(gray, rect)
    # convert shape object to numpy array format
    shape = face_utils.shape_to_np(shape)

    # convert dlib's rectangle to OpenCV-style bounding-box
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # draw landmarks on face
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", image)
cv2.waitKey(0)