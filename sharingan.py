"""Detect eyes and transform sharingan effect."""
# USAGE
# python sharingan.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/elon.jpg --eye images/sharingan.png
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
    help="path to the input image")
ap.add_argument("-m", "--eye", required=True,
	help="path to the sharineye image")
args = vars(ap.parse_args())

# init dlib's HOG face detector and the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
face_hog_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(args["shape_predictor"])

# read input image and resize it
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)

# the image is made up of 4 layers: Blue, Green, Red, and an Alpha transparency layer (BGR-A)
img_eye = cv2.imread(args["eye"], -1)
# take just the alpha layer used for masking the area for the glasses
original_mask_eye = img_eye[:, :, 3]
# take the inverse mask for region around the glasses
original_mask_eye_inv = cv2.bitwise_not(original_mask_eye)
img_eye = img_eye[:, :, 0:3]
original_eye_height, original_eye_width = img_eye.shape[:2]

# detect faces in input image
rects = face_hog_detector(image, 1)

for (i, rect) in enumerate(rects):
    shape = landmark_predictor(image, rect)
    
    left_eye_width = abs(shape.part(37).x - shape.part(40).x) * 2
    left_eye_height = int(left_eye_width * original_eye_height / original_eye_width)
    left_eye = cv2.resize(img_eye, (left_eye_width, left_eye_height), interpolation=cv2.INTER_AREA)
    mask_left_eye = cv2.resize(original_mask_eye, (left_eye_width, left_eye_height), interpolation=cv2.INTER_AREA)
    mask_left_eye_inv = cv2.resize(original_mask_eye_inv, (left_eye_width, left_eye_height), interpolation=cv2.INTER_AREA)

    right_eye_width = abs(shape.part(43).x - shape.part(46).x) * 2
    right_eye_height = int(right_eye_width * original_eye_height / original_eye_width)
    right_eye = cv2.resize(img_eye, (right_eye_width, right_eye_height))
    mask_right_eye = cv2.resize(original_mask_eye, (right_eye_width, right_eye_height), interpolation=cv2.INTER_AREA)
    mask_right_eye_inv = cv2.resize(original_mask_eye_inv, (right_eye_width, right_eye_height), interpolation=cv2.INTER_AREA)

    y1_l = int(shape.part(39).y-10)
    y2_l = int(y1_l + left_eye_height)
    x1_l = int(shape.part(37).x)
    x2_l = int(x1_l + left_eye_width)
    roi_l = image[y1_l:y2_l, x1_l:x2_l]
    roi_l_bg = cv2.bitwise_and(roi_l, roi_l, mask=mask_left_eye_inv)
    roi_l_fg = cv2.bitwise_and(left_eye, left_eye, mask=mask_left_eye)
    image[y1_l:y2_l, x1_l:x2_l] = cv2.add(roi_l_bg, roi_l_fg)

    y1_r = int(shape.part(45).y-10)
    y2_r = int(y1_r + right_eye_height)
    x1_r = int(shape.part(43).x)
    x2_r = int(x1_r + right_eye_width)
    roi_r = image[y1_r:y2_r, x1_r:x2_r]
    roi_r_bg = cv2.bitwise_and(roi_r, roi_r, mask=mask_right_eye_inv)
    roi_r_fg = cv2.bitwise_and(right_eye, right_eye, mask=mask_right_eye)
    image[y1_r:y2_r, x1_r:x2_r] = cv2.add(roi_r_bg, roi_r_fg)

cv2.imshow("Output", image)
cv2.waitKey(0)