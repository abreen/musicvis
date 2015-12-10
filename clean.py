import numpy
import cv2

import params
import util


def find_staff_lines(img):
    """Use a Hough line transform and (other methods) to detect staff lines,
    returning a list of the line endpoints along with a binary image in which
    the staff lines have been removed.
    """

    # convert to grayscale, invert, and threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.bitwise_not(img_gray, img_gray)
    rv, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

    """
    util.show('Inverted', img_gray)
    """

    # use the probablistic Hough transform to detect staff lines (or parts of staff lines)
    lines = cv2.HoughLinesP(
                img_gray,
                1,                              # HOUGH_DISTANCE_RESOLUTION - it seems this is very important for detecting the thickness of staff lines
                numpy.pi / 270,                 # HOUGH_ANGLE_RESOLUTION,
                100,                            # HOUGH_THRESHOLD,
                minLineLength=100,              # HOUGH_MIN_LINE_LENGTH
                maxLineGap=10                   # HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        util.debug('detected no lines')
        return
    else:
        lines = lines[0].tolist()
        util.debug('detected %d lines: %s' % (len(lines), str(lines)))

    rows, cols = img.shape[:2]
    axes = [(0, 0, cols, 0),
            (cols, 0, cols, rows),
            (0, rows, cols, rows),
            (0, 0, 0, rows)]

    # discard bad candidates for staff lines
    thresh_dist = 4                         # throw out lines within 4 pixels of axes
    thresh_angl = 2 * (numpy.pi / 180)      # radians; throw out lines which have angle greater than this many degrees

    def valid(line):
        """Return True if a line is valid: if it is within the angle threshold
        (see above) and it is not too close to an axis (see above).
        """
        x1, y1, x2, y2 = line
        if util.angle(line, axes[0]) > thresh_angl:
            return False
        else:
            for axis in axes:
                if util.distance(line, axis) < thresh_dist:
                    return False
        return True

    lines = [line for line in lines if valid(line)]

    # next: calculate y-coordinates of each staff line via weighted averaging (centroids)

    staff_lines = []

    # create a binary image containing only the detected staff lines
    rows, cols = img.shape[:2]

    temp = numpy.zeros((rows, cols, 1), numpy.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(temp, (x1, y1), (x2, y2), 255, 1)   # for pretty purposes

    contour_finder = temp.copy()
    contours, hierarchy = cv2.findContours(contour_finder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # find the centroid of each staff line using moments
    for contour in contours:
        r, c, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(temp[c:c+h, r:r+w])

        xbar = moments['m10'] / moments['m00']
        ybar = moments['m01'] / moments['m00']

        staff_lines.append(c + ybar)
        cv2.circle(temp, (int(r + xbar), int(c + ybar)), 1, util.RED)
        cv2.rectangle(temp, (r, c), (r + w, c + h), 127, 1)

    util.debug('kept %d lines: %s' % (len(lines), str(lines)))

    """
    img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in lines:
        cv2.line(img_color, (x1, y1), (x2, y2), util.RED, 1)   # for pretty purposes

    util.show('Staff lines', img_color, True)
    """

    # staff line removal: draw over binary image and perform morphological operations
    for x1, y1, x2, y2 in lines:
        # zero out the staff lines, restore important pixels later
        cv2.line(img_gray, (x1, y1), (x2, y2), 0, 1, 4)

    kern4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))

    img_gray = cv2.erode(img_gray, kern2, iterations=1)
    img_gray = cv2.dilate(img_gray, kern2, iterations=1)

    img_gray = cv2.dilate(img_gray, kern4, iterations=1)
    img_gray = cv2.erode(img_gray, kern4, iterations=1)

    """
    util.show('After morphological operations', img_gray, True)
    """

    return img_gray, staff_lines

