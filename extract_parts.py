"""Extract facial parts using facial landmarks."""
# USAGE
# python extract_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/elon.jpg

import cv2
import dlib
import argparse
import imutils
import numpy as np
from imutils import face_utils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

# init dlib's HOG face detector and facial landmark predictor
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

    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # clone original image to draw on it
        clone = image.copy()
        # display name of the face part
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # draw specific face part
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # extract the ROI of the face region as separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y: y+h, x: x+w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # show the particular face part
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)