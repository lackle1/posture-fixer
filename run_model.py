from joblib import load
import cv2
import sys
import keras
import numpy as np

model = load('classification_model.joblib')

# Initialise camera and video
cv2.namedWindow('Camera')
vc = cv2.VideoCapture(0)

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

    square_img = img[top:bottom, left:right]
    return cv2.resize(square_img, (256, 256))

def make_prediction(frame, model):
    prediction = model.predict(frame, verbose=0).argmax(axis=1)
    return prediction[0]

# Runs ~50 times per second
current_time = timer
while result:
    if timer != -1:
        current_time -= 1

    cv2.imshow('Camera', frame)
    result, frame = vc.read()
    frame = process_image(frame)
    pred = make_prediction(np.array([frame]), model)
    frame = cv2.putText(frame, f'{pred}', (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    key = cv2.waitKey(20)
    if key == 27: # Exit on ESC
        break

vc.release()
cv2.destroyWindow('Camera')