# Testing HOG detector and CNN based detector using Dlib
import time
import cv2
import dlib
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# init dlib detectors
hog_face_detector = dlib.get_frontal_face_detector()
cnn_face_detector = dlib.cnn_face_detection_model_v1('cnn_weights.dat')

# hog detector testing
start = time.time()
faces_hog = hog_face_detector(image, 1)
end = time.time()
print("HOG + SVM Execution time: " + str(end-start))

# draw bounding boxes
for face in faces_hog:
    x = face.left()
    y = face.top()
    w = face.right() - x
    h = face.bottom() - y

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)

# cnn detector testing
start = time.time()
faces_cnn = cnn_face_detector(image, 1)
end = time.time()
print("CNN Execution time: " + str(end-start))

# draw bounding boxes
for face in faces_cnn:
    x = face.rect.left()
    y = face.rect.top()
    w = face.rect.right() - x
    h = face.rect.bottom() - y

    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("image", image)
cv2.waitKey(0)