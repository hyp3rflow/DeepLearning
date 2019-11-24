import numpy as np
import cv2 as cv
import sys
import os
from copy import deepcopy
sys.path.append(os.pardir)

faceCascade = cv.CascadeClassifier(
    "/Users/katchoflowy/flow/DeepLearning/OpenCV/haarcascade_frontalcatface_extended.xml")


def detectCat():
    img = cv.imread(
        "/Users/katchoflowy/flow/DeepLearning/OpenCV/Images/img1.jpeg", cv.IMREAD_COLOR)
    tmp = deepcopy(img)
    grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        grayImg, scaleFactor=1.001, minNeighbors=1, minSize=(100, 100))

    num = 0

    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_cat = tmp[y: y + h, x: x + w]
        cv.imwrite(str(num) + ".png", img_cat)
        num += 1

    return img


img = detectCat()

if len(img) != 0:
    img = cv.resize(img, None, fx=0.3, fy=0.3, interpolation=cv.INTER_AREA)
    cv.imshow('Face', img)

cv.waitKey(0)
cv.destroyWindow('Face')
