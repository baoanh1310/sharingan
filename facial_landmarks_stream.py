"""Real-time facial landmark detection in video stream."""
# USAGE
# python facial_landmarks_stream.py --shape-predictor shape_predictor_68_face_landmarks.dat

import datetime
import argparse
import imutils
import time
import dlib
import cv2
from imutils.video import VideoStream
from imutils import face_utils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
args = vars(ap.parse_args())

# init dlib's HOG face detector and the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
face_hog_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(args["shape_predictor"])

vs = VideoStream().start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in grayscale frame
    rects = face_hog_detector(gray, 0)

    for rect in rects:
        shape = landmark_predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

cv2.destroyAllWindows()
vs.stop()