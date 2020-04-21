"""Detect eyes and transform sharingan effect in video stream."""
# USAGE
# python sharingan_stream.py --shape-predictor shape_predictor_68_face_landmarks.dat --eye images/sharingan.png
from imutils.video import VideoStream
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
ap.add_argument("-m", "--eye", required=True,
    help="path to the sharineye image")
args = vars(ap.parse_args())

# init dlib's HOG face detector and the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
face_hog_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(args["shape_predictor"])

# the image is made up of 4 layers: Blue, Green, Red, and an Alpha transparency layer (BGR-A)
img_eye = cv2.imread(args["eye"], -1)
# take just the alpha layer used for masking the area for the glasses
original_mask_eye = img_eye[:, :, 3]
# take the inverse mask for region around the glasses
original_mask_eye_inv = cv2.bitwise_not(original_mask_eye)
img_eye = img_eye[:, :, 0:3]
original_eye_height, original_eye_width = img_eye.shape[:2]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = face_hog_detector(gray, 0)

    for rect in rects:
        shape = landmark_predictor(gray, rect)
        
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
        roi_l = frame[y1_l:y2_l, x1_l:x2_l]
        roi_l_bg = cv2.bitwise_and(roi_l, roi_l, mask=mask_left_eye_inv)
        roi_l_fg = cv2.bitwise_and(left_eye, left_eye, mask=mask_left_eye)
        frame[y1_l:y2_l, x1_l:x2_l] = cv2.add(roi_l_bg, roi_l_fg)

        y1_r = int(shape.part(45).y-10)
        y2_r = int(y1_r + right_eye_height)
        x1_r = int(shape.part(43).x)
        x2_r = int(x1_r + right_eye_width)
        roi_r = frame[y1_r:y2_r, x1_r:x2_r]
        roi_r_bg = cv2.bitwise_and(roi_r, roi_r, mask=mask_right_eye_inv)
        roi_r_fg = cv2.bitwise_and(right_eye, right_eye, mask=mask_right_eye)
        frame[y1_r:y2_r, x1_r:x2_r] = cv2.add(roi_r_bg, roi_r_fg)


    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()