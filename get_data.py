import sys
import os
import cv2

folder = f'data/{sys.argv[1]}'
if os.path.isdir(folder):
    img_index = len(os.listdir(folder))
    print(f'Last image index: {img_index}')
else:
    os.mkdir(folder)
    img_index = 0

# Initialise camera and video
cv2.namedWindow('Camera')
vc = cv2.VideoCapture(1)

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

# Runs ~50 times per second
current_time = timer
while result:
    if timer != -1:
        current_time -= 1

    cv2.imshow('Camera', frame)
    result, frame = vc.read()
    frame = process_image(frame)
    key = cv2.waitKey(20)
    if current_time == 0 or key == 32: # Save photo if SPACE
        filepath = f'{folder}/img_{img_index}.png'
        cv2.imwrite(filepath, frame)
        img_index += 1
        current_time = timer
        print(filepath + ' saved.')
    elif key == 27: # Exit on ESC
        break

vc.release()
cv2.destroyWindow('Camera')