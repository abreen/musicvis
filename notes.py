import numpy
import cv2

import params
import util

def find_staff(img):
    """This function uses a Hough line transform to detect the five lines which
    comprise a staff in a given image.
    """

    # Convert to grayscale, invert, and threshold
    edge_map_bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.bitwise_not ( edge_map_bw, edge_map_bw )
    rv, edge_map_bw = cv2.threshold (edge_map_bw, 127, 255, cv2.THRESH_BINARY)

    """
    util.show('Edges', edge_map_bw)
    """

    # Use the probablistic Hough transform to detect staff lines (or parts of staff lines)
    lines = cv2.HoughLinesP(
                edge_map_bw,
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

    # Determine if lines are valid staff lines
    thresh_dist = 4                         # throw out lines within 4 pixels of axes
    thresh_angl = 2 * (numpy.pi / 180)      # radians; throw out lines which have angle greater than this many degrees
    
    def valid(line):
        x1, y1, x2, y2 = line
        if util.angle(line, axes[0]) > thresh_angl:
            return False
        else:
            for axis in axes:
                if util.distance(line, axis) < thresh_dist:
                    return False
        return True

    lines = [line for line in lines if valid(line)]


    # Calculate y-coordinates of each staff line via weighted averaging (centroids)
    staves = []

    # Create a binary image containing only the detected staff lines
    rows, cols = edge_map_bw.shape[:2]
    temp = numpy.zeros((rows, cols, 1), numpy.uint8)
    for x1, y1, x2, y2 in lines:
        cv2.line(temp, (x1, y1), (x2, y2), 255, 1)   # for pretty purposes

    contour_finder = temp.copy()
    contours, hierarchy = cv2.findContours(contour_finder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Find the centroid of each staff line using moments
    for contour in contours:
        r, c, w, h = cv2.boundingRect(contour)
        moments = cv2.moments(temp[c:c+h, r:r+w])

        xbar = (moments['m10']/moments['m00'])
        ybar = (moments['m01']/moments['m00'])

        staves.append(c + ybar)
        cv2.circle(temp, ((int) (r + xbar), (int) (c + ybar)), 1, util.RED)
        cv2.rectangle(temp, (r, c), (r+w, c+h), 127, 1)


    edge_map_color = cv2.cvtColor(edge_map_bw, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in lines:
        cv2.line(edge_map_color, (x1, y1), (x2, y2), util.RED, 1)   # for pretty purposes
        # cv2.line(edge_map_color, (x1, y1), (x2, y2), util.RED, params.LINE_THICKNESS) <- this makes our results look deceptively better?

    util.debug('kept %d lines: %s' % (len(lines), str(lines)))

    """ Pretty picture showing staff detection
    util.show('Staves', edge_map_color, True)
    """


    # Staff line removal: draw over binary image and perform morphological operations
    for x1, y1, x2, y2 in lines:
        cv2.line(edge_map_bw, (x1, y1), (x2, y2), 0, 1, 4)         # zero out the staff lines, restore important pixels later

    kern4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))

    edge_map_bw = cv2.erode(edge_map_bw, kern2, iterations=1)
    edge_map_bw = cv2.dilate(edge_map_bw, kern2, iterations=1)

    edge_map_bw = cv2.dilate(edge_map_bw, kern4, iterations=1)
    edge_map_bw = cv2.erode(edge_map_bw, kern4, iterations=1)


    """ Show the image after staff removal
    util.show('Morphed', edge_map_bw, True)
    """


    contour_finder = edge_map_bw.copy()
    contours, hierarchy = cv2.findContours(contour_finder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # OK here are rectangles, they're totally prettier
    musical_objects = []
    for contour in contours:
        musical_objects.append(cv2.boundingRect(contour))

    util.debug('found %d musical objects: %s' % (len(musical_objects), str(musical_objects)))

    edge_map_color = cv2.cvtColor(edge_map_bw, cv2.COLOR_GRAY2BGR)
    for r, c, w, h in musical_objects:
        cv2.rectangle(edge_map_color, (r, c), (r+w, c+h), util.RED, 1)

        img2 = cv2.GaussianBlur(edge_map_bw, (25, 25), 0)
        util.show('blurrrrred', img2)

        # Hough method of circle detection not so good on notes
        circles = cv2.HoughCircles(img2[c:c+h, r:r+w], cv2.cv.CV_HOUGH_GRADIENT, 1, 10, param1=50,param2=15)
        if circles is not None:
            for x, y, rad in circles[0]:
                cv2.circle(edge_map_color, ((int)(r+x), (int)(c+y)), rad, util.RED)



    """ Show the segmented image
    """
    util.show('Segmented', edge_map_color, True)


    # error checking? merge disconnected components?
    return staves

