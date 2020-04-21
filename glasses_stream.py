"""Apply glasses filter to video stream."""
# USAGE
# python glasses_stream.py --shape-predictor shape_predictor_68_face_landmarks.dat --glass images/glasses.png
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
ap.add_argument("-g", "--glass", required=True, help="path to the glasses image")
args = vars(ap.parse_args())

# init dlib's HOG face detector and the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
face_hog_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(args["shape_predictor"])

# the image is made up of 4 layers: Blue, Green, Red, and an Alpha transparency layer (BGR-A)
img_glasses = cv2.imread(args["glass"], -1)
# take just the alpha layer used for masking the area for the glasses
original_mask_glasses = img_glasses[:, :, 3]
# take the inverse mask for region around the glasses
original_mask_glasses_inv = cv2.bitwise_not(original_mask_glasses)
img_glasses = img_glasses[:, :, 0:3]
original_glasses_height, original_glasses_width = img_glasses.shape[:2]

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

        glasses_width = abs(shape.part(16).x - shape.part(1).x)
        glasses_height = int(glasses_width * original_glasses_height / original_glasses_width) 
       
        glasses = cv2.resize(img_glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
        mask_glasses = cv2.resize(original_mask_glasses, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
        mask_glasses_inv = cv2.resize(original_mask_glasses_inv, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)
        y1 = int(shape.part(24).y)
        y2 = int(y1 + glasses_height)
        x1 = int(shape.part(27).x - (glasses_width/2))
        x2 = int(x1 + glasses_width)
        roi = frame[y1:y2, x1:x2]
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_glasses_inv)
        roi_fg = cv2.bitwise_and(glasses, glasses, mask=mask_glasses)
        frame[y1:y2, x1:x2] = cv2.add(roi_bg, roi_fg)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()