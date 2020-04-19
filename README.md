# Introduction

In this pet project, I use dlib to predict facial landmarks, then extract eyes from face and finally make transform the eyes into special eyes in Naruto anime.

# Installation

```bash
pip install -r requirements.txt
git clone https://github.com/baoanh1310/sharingan.git
```

# How to run?

## 1. If you want to predict facial landmarks from input image
```bash
python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/sample.jpg
```

## 2. If you want to predict facial landmark in real-time video stream
```bash
python facial_landmarks_stream.py --shape-predictor shape_predictor_68_face_landmarks.dat
```

## 3. If you want to extract face parts from input image
```bash
python extract_parts.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/elon.jpg
```