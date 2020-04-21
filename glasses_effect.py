"""Add glasses effect on faces detected in input image."""
# USAGE
# python glasses_effect.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/elon.jpg --glass images/glasses.png

import numpy as np
import cv2 
import dlib
import argparse
import glob
import imutils
from imutils import face_utils

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-g", "--glass", required=True, help="path to the glasses image")
args = vars(ap.parse_args())

# init dlib's HOG face detector and facial landmark predictor
face_hog_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


# load the input image
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

# the image is made up of 4 layers: Blue, Green, Red, and an Alpha transparency layer (BGR-A)
img_glasses = cv2.imread(args["glass"], -1)
# take just the alpha layer used for masking the area for the glasses
original_mask_glasses = img_glasses[:, :, 3]
# take the inverse mask for region around the glasses
original_mask_glasses_inv = cv2.bitwise_not(original_mask_glasses)
img_glasses = img_glasses[:, :, 0:3]
original_glasses_height, original_glasses_width = img_glasses.shape[:2]

# detect faces in input image
rects = face_hog_detector(image, 1)
for (i, rect) in enumerate(rects):
    shape = landmark_predictor(image, rect)
    
    glasses_width = abs(shape.part(16).x - shape.part(1).x)
    glasses_height = int(glasses_width * original_glasses_height / original_glasses_width) 
   
    glasses = cv2.resize(img_glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
    mask_glasses = cv2.resize(original_mask_glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
    mask_glasses_inv = cv2.resize(original_mask_glasses_inv, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
    y1 = int(shape.part(24).y)
    y2 = int(y1 + glasses_height)
    x1 = int(shape.part(27).x - (glasses_width/2))
    x2 = int(x1 + glasses_width)
    roi = image[y1:y2, x1:x2]
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_glasses_inv)
    roi_fg = cv2.bitwise_and(glasses, glasses, mask=mask_glasses)
    image[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

cv2.imshow("Output", image)
cv2.waitKey(0)
