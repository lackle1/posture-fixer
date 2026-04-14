from joblib import load
import cv2
import sys
import keras
import numpy as np

# import arduino_communication
import time

model = load('classification_model.joblib')

# Initialise camera and video
cv2.namedWindow('Camera')
vc = cv2.VideoCapture() # 0 if using own laptop webcam, 1 if using attached webcam

if vc.isOpened():
    result, frame = vc.read()
else:
    result = False

timer = 50

def process_image(img):
    h, w = img.shape[:2]
    min_dim = min(h, w)

    top = (h - min_dim) // 2
    bottom = top + min_dim
    left = (w - min_dim) // 2
    right = left + min_dim

    img = img[top:bottom, left:right]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=2)
    return cv2.resize(img, (256, 256))

def make_prediction(frame, model):
    data = np.array([cv2.resize(frame, (64, 64))])
    prediction = model.predict(data, verbose=0)[0]
    return prediction

# Runs ~50 times per second
current_time = timer
text = ''
while result:
    current_time -= 1

    result, frame = vc.read()
    frame = process_image(frame)

    if current_time < 0:
        pred = make_prediction(frame, model)
        index = pred.argmax()
        text = f'{index}'
        # arduino_communication.doSomething(index)
        current_time = timer

    frame = cv2.putText(frame, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(20)
    if key == 27: # Exit on ESC
        break

vc.release()
cv2.destroyWindow('Camera')