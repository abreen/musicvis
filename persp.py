import numpy
import cv2

import util


BLUR_RADIUS = 13
CANNY_THRESHOLD_LOW = 300
CANNY_THRESHOLD_HIGH = 400
HOUGH_DISTANCE_RESOLUTION = 2               # pixels
HOUGH_ANGLE_RESOLUTION = numpy.pi / 270     # radians
HOUGH_THRESHOLD = 100
HOUGH_MIN_LINE_LENGTH = 250                 # pixels
HOUGH_MAX_LINE_GAP = 10                     # pixels
ANGLE_THRESHOLD = 2 * (numpy.pi / 180)      # radians
LINE_THICKNESS = 2                          # pixels


def find_page(img):
    """This function uses a Hough line transform to detect the four sides of a
    piece of paper in the image, applies an inverse warp transformation to
    correct for perspective distortion, returning the corrected image.
    """
    rows, cols = img.shape[:2]

    img2 = cv2.GaussianBlur(img, (BLUR_RADIUS, BLUR_RADIUS), 0)

    """
    util.show('Blurred', img2)
    """

    # use Canny edge deduction to obtain an edge map
    edge_map_bw = cv2.Canny(img2, CANNY_THRESHOLD_LOW, CANNY_THRESHOLD_HIGH)

    """
    util.show('Edges', edge_map_bw)
    """

    # use the probablistic Hough transform
    lines = cv2.HoughLinesP(
                edge_map_bw,
                HOUGH_DISTANCE_RESOLUTION,
                HOUGH_ANGLE_RESOLUTION,
                HOUGH_THRESHOLD,
                minLineLength=HOUGH_MIN_LINE_LENGTH,
                maxLineGap=HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        util.debug('detected no lines')
        return
    else:
        lines = lines[0].tolist()
        util.debug('detected %d lines: %s' % (len(lines), str(lines)))

    edge_map_color = cv2.cvtColor(edge_map_bw, cv2.COLOR_GRAY2BGR)

    while True:
        for a in lines:
            util.debug('current line: %s' % str(a))
            too_similar = False

            for b in lines:
                if a == b:
                    continue

                t = _angle(a, b)

                util.debug('other line: %s (angle: %f)' % (str(b), t))

                if t < ANGLE_THRESHOLD:
                    too_similar = True
                    break

            if too_similar:
                util.debug('removing %s' % str(a))
                lines.remove(a)
                break

        if not too_similar:
            break

    util.debug('kept %d lines: %s' % (len(lines), str(lines)))

    if len(lines) != 4:
        return None

    for x1, y1, x2, y2 in lines:
        cv2.line(edge_map_color, (x1, y1), (x2, y2), util.RED, LINE_THICKNESS)

    """
    util.show('Lines', edge_map_color)
    """

    corners = set()
    for a in lines:
        for b in lines:
            if a == b:
                continue

            i = _intersection(a, b)
            if i[0] >= 0 and i[0] < cols and i[1] >= 0 and i[1] < cols:
                corners.add(i)

    # take the most extreme four corners to be the corners of the page

    asc_by_x = lambda seq: sorted(seq, cmp=lambda a, b: int(a[0] - b[0]))
    asc_by_y = lambda seq: sorted(seq, cmp=lambda a, b: int(a[1] - b[1]))
    dsc_by_x = lambda seq: sorted(seq, cmp=lambda a, b: int(b[0] - a[0]))
    dsc_by_y = lambda seq: sorted(seq, cmp=lambda a, b: int(b[1] - a[1]))

    ul = asc_by_y(asc_by_x(corners))[0]
    corners.remove(ul)
    ur = asc_by_y(dsc_by_x(corners))[0]
    corners.remove(ur)
    ll = dsc_by_y(asc_by_x(corners))[0]
    corners.remove(ll)
    lr = dsc_by_y(dsc_by_x(corners))[0]
    corners.remove(lr)

    for x, y in [ul, ur, ll, lr]:
        cv2.circle(edge_map_color, (x, y), 3, util.GREEN, -1)

    """
    util.show('Lines & corners', edge_map_color)
    """

    # use these four corners to construct a transformation matrix
    rows2, cols2 = 800, 1200

    m = cv2.getPerspectiveTransform(
            numpy.array([ul, ur, ll, lr]).astype('float32'),
            numpy.array(
                [
                    [0, 0],
                    [cols2 - 1, 0],
                    [cols2 - 1, rows2 - 1],
                    [0, rows2 - 1]
                ]
            ).astype('float32')
    )

    corrected = cv2.warpPerspective(img, m, (cols2, rows2))

    """
    util.show('Corrected', corrected)
    """

    return corrected


def _intersection(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b

    c = ((x1-x2) * (y3-y4)) - ((y1-y2) * (x3-x4))
    if c == 0:
        return (0, 0)

    return (
        ((x1*y2 - y1*x2) * (x3-x4) - (x1-x2) * (x3*y4 - y3*x4)) / c,
        ((x1*y2 - y1*x2) * (y3-y4) - (y1-y2) * (x3*y4 - y3*x4)) / c
    )


def _angle(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b

    m1 = float(y2 - y1) / float(x2 - x1)
    m2 = float(y4 - y3) / float(x4 - x3)

    return numpy.arctan(numpy.abs((m1 - m2) / (1 + (m1 * m2))))
