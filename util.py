import numpy
import cv2
import os

DEBUG = True
MAX_WIDTH = 800
MAX_HEIGHT = 600
SHRINK_METHOD = cv2.INTER_AREA
RED = (0, 0, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def debug(msg):
    if DEBUG:
        print(msg)


def show(title, img, save=False):
    rows, cols = img.shape[:2]

    if not (rows < MAX_HEIGHT and cols < MAX_WIDTH):
        if rows > MAX_HEIGHT:
            # resize to fit height
            factor = float(MAX_HEIGHT) / rows
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=SHRINK_METHOD)

        rows, cols = img.shape[:2]

        if cols > MAX_WIDTH:
            # resize to fit width
            factor = float(MAX_WIDTH) / cols
            img = cv2.resize(img, None, fx=factor, fy=factor, interpolation=SHRINK_METHOD)

    if save:
        # print("saving " + title + "....")
        if not os.path.isdir("output"):
            os.mkdir("output")

        cv2.imwrite("output/" + title + ".png", img)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)


def intersection(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b

    c = ((x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4))
    if c == 0:
        return (0, 0)

    return (
        ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / c,
        ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / c
    )


def angle(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b

    m1 = float(y2 - y1) / float(x2 - x1)
    m2 = float(y4 - y3) / float(x4 - x3)

    return numpy.arctan(numpy.abs((m1 - m2) / (1 + (m1 * m2))))


def distance(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b

    def point_distance(x1, y1, x2, y2):
        x_diff = x2 - x1
        y_diff = y2 - y1
        return numpy.sqrt((x_diff * x_diff) + (y_diff * y_diff))

    return min(
            point_distance(x1, y1, x3, y3), point_distance(x2, y2, x4, y4),
            point_distance(x1, y1, x4, y4), point_distance(x2, y2, x3, y3),
    )
