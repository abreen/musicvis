import cv2


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


def show(title, img):
    rows, cols = img.shape[:2]

    if rows < MAX_HEIGHT and cols < MAX_WIDTH:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img)
        return

    img2 = img.copy()

    if rows > MAX_HEIGHT:
        # resize to fit height
        factor = float(MAX_HEIGHT) / rows
        img2 = cv2.resize(img, None, fx=factor, fy=factor, interpolation=SHRINK_METHOD)

    rows, cols = img2.shape[:2]

    if cols > MAX_WIDTH:
        # resize to fit width
        factor = float(MAX_WIDTH) / cols
        img2 = cv2.resize(img2, None, fx=factor, fy=factor, interpolation=SHRINK_METHOD)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img2)
