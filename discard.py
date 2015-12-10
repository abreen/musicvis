import numpy
import cv2

import params
import util


def filter_musical_objects(img, objs):
    objs2 = []

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for r, c, w, h in objs:
        cv2.rectangle(img_color, (r, c), (r+w, c+h), util.RED, 1)

        img2 = cv2.GaussianBlur(img, (29, 29), 0)

        circles = cv2.HoughCircles(img2[c:c+h, r:r+w], cv2.cv.CV_HOUGH_GRADIENT, 1, 10, param1=50,param2=16)
        if circles is not None and len(circles[0]) == 1:
            objs2.append((r, c, w, h))

            for x, y, rad in circles[0]:
                cv2.circle(img_color, ((int)(r+x), (int)(c+y)), rad, util.RED)

    util.show('Circles', img_color)

    return objs2
