import numpy
import cv2

import params
import util


def find_page(img):
    """This function uses a Hough line transform to detect the four sides of a
    piece of paper in the image, applies an inverse warp transformation to
    correct for perspective distortion, returning the corrected image.
    """
    rows, cols = img.shape[:2]

    img2 = cv2.GaussianBlur(img, (params.BLUR_RADIUS, params.BLUR_RADIUS), 0)

    """
    util.show('Blurred', img2)
    """

    # use Canny edge deduction to obtain an edge map
    edge_map_bw = cv2.Canny(img2, params.CANNY_THRESHOLD_LOW, params.CANNY_THRESHOLD_HIGH)

    """
    util.show('Edges', edge_map_bw)
    """

    # use the probablistic Hough transform
    lines = cv2.HoughLinesP(
                edge_map_bw,
                params.HOUGH_DISTANCE_RESOLUTION,
                params.HOUGH_ANGLE_RESOLUTION,
                params.HOUGH_THRESHOLD,
                minLineLength=params.HOUGH_MIN_LINE_LENGTH,
                maxLineGap=params.HOUGH_MAX_LINE_GAP
    )

    if lines is None:
        util.debug('detected no image outlines')
        return
    else:
        lines = lines[0].tolist()
        util.debug('detected %d image outlines: %s' % (len(lines), str(lines)))

    edge_map_color = cv2.cvtColor(edge_map_bw, cv2.COLOR_GRAY2BGR)

    while True:
        for a in lines:
            """
            util.debug('current line: %s' % str(a))
            """
            too_similar = False

            for b in lines:
                if a == b:
                    continue

                t = util.angle(a, b)
                d = util.distance(a, b)

                """
                util.debug('other line: %s (angle: %f)' % (str(b), t))
                """

                if d < params.DISTANCE_THRESHOLD and t < params.ANGLE_THRESHOLD:
                    too_similar = True
                    break

            if too_similar:
                """
                util.debug('removing %s' % str(a))
                """
                lines.remove(a)
                break

        if not too_similar:
            break

    util.debug('kept %d image outlines: %s' % (len(lines), str(lines)))

    for x1, y1, x2, y2 in lines:
        cv2.line(edge_map_color, (x1, y1), (x2, y2), util.RED, params.LINE_THICKNESS)

    """
    util.show('Lines', edge_map_color)
    """

    if len(lines) != 4:
        return None

    corners = set()
    for a in lines:
        for b in lines:
            if a == b:
                continue

            i = util.intersection(a, b)
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

    # for the height/width of the new image, we use a bounding box
    rows2 = max(ll[1], lr[1]) - min(ul[1], ur[1])
    cols2 = max(ur[0], lr[0]) - min(ul[0], ll[0])

    """
    print("source: ", numpy.array([ul, ur, ll, lr]).astype('float32'))
    print("dest: ", numpy.array(
                [
                    [0, 0],
                    [cols2 - 1, 0],
                    [cols2 - 1, rows2 - 1],
                    [0, rows2 - 1]
                ]
            ).astype('float32'))
    """

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

