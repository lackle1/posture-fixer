from joblib import load
import cv2
import sys
import keras
import numpy as np

import arduino_communication
import time

model = load('classification_model.joblib')

# Initialise camera and video
cv2.namedWindow('Camera')
vc = cv2.VideoCapture(0) # 0 if using own laptop webcam, 1 if using attached webcam

if vc.isOpened():
    result, frame = vc.read()
else:
    result = False

# Use timer if it was passed in sys args
if len(sys.argv) == 3:
    timer = int(sys.argv[2])
else:
    timer = -1

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
while result:
    if timer != -1:
        current_time -= 1

    result, frame = vc.read()
    frame = process_image(frame)
    pred = make_prediction(frame, model)
    index = pred.argmax()
    text = f'{index}'
    frame = cv2.putText(frame, text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Camera', frame)

    key = cv2.waitKey(20)
    if key == 27: # Exit on ESC
        break

    arduino_communication.doSomething(index)
    time.sleep(1)
    # if index != 0:
    #     time.sleep(2)

vc.release()
cv2.destroyWindow('Camera')