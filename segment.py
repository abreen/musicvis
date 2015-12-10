import numpy
import cv2

import params
import util


def find_objects(img):
    img2 = img.copy()
    contours, hierarchy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    objs = []
    for contour in contours:
        objs.append(cv2.boundingRect(contour))

    return objs
