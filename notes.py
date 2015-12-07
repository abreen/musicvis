import numpy
import cv2

import params
import util

def find_staff(img):
    """This function uses a Hough line transform to detect the five lines which
    comprise a staff in a given image.
    """

    # blur_radius_staff = 1

    # img2 = cv2.GaussianBlur(img, (blur_radius_staff, blur_radius_staff), 0)

    # # use Canny edge deduction to obtain an edge map
    # edge_map_bw = cv2.Canny(img2, params.CANNY_THRESHOLD_LOW, params.CANNY_THRESHOLD_HIGH)

    # Alternatively, convert to grayscale, invert, and threshold
    edge_map_bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.bitwise_not ( edge_map_bw, edge_map_bw )
    rv, edge_map_bw = cv2.threshold (edge_map_bw, 127, 255, cv2.THRESH_BINARY)

    """
    util.show('Edges', edge_map_bw)
    """
    # use the probablistic Hough transform
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

    # Calculate y-coordinates of each staff line via weighted averaging
    staves = []

    centers = []
    numbers = []
    # for x1, y1, x2, y2 in lines:
    #     if y1 != y2:
    #         util.debug("probably not what you want, check notes.py:72")
    #     centers.append(y1)
    #     numbers.append(x2-x1)

    # for center in centers:
    #     if 


    edge_map_color = cv2.cvtColor(edge_map_bw, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in lines:
        cv2.line(edge_map_color, (x1, y1), (x2, y2), util.RED, 1)   # for pretty purposes
        # cv2.line(edge_map_color, (x1, y1), (x2, y2), util.RED, params.LINE_THICKNESS) <- this makes our results look deceptively better?

    util.debug('kept %d lines: %s' % (len(lines), str(lines)))


    """ Pretty picture showing staff detection
    util.show('Staves', edge_map_color)
    """
    util.show('Staves', edge_map_color, True)

    # Working method #1:
    # for x1, y1, x2, y2 in lines:
    #     cv2.line(edge_map_bw, (x1, y1), (x2, y2), 0, 2)         # zero out the staff lines, restore important pixels later 
    # kern3 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))
    # kern4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))

    # edge_map_bw = cv2.dilate(edge_map_bw, kern4, iterations=1)
    # edge_map_bw = cv2.dilate(edge_map_bw, kern3, iterations=1)
    # edge_map_bw = cv2.erode(edge_map_bw, kern3, iterations=1)
    # edge_map_bw = cv2.erode(edge_map_bw, kern4, iterations=1)

    # This seems to work slightly better:
    for x1, y1, x2, y2 in lines:
        cv2.line(edge_map_bw, (x1, y1), (x2, y2), 0, 1, 4)         # zero out the staff lines, restore important pixels later

    kern4 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,4))
    kern2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))

    edge_map_bw = cv2.erode(edge_map_bw, kern2, iterations=1)
    edge_map_bw = cv2.dilate(edge_map_bw, kern2, iterations=1)

    edge_map_bw = cv2.dilate(edge_map_bw, kern4, iterations=1)
    edge_map_bw = cv2.erode(edge_map_bw, kern4, iterations=1)


    """ Show the image after staff removal
    util.show('Morphed', edge_map_bw)
    """
    util.show('Morphed', edge_map_bw, True)

    contour_finder = edge_map_bw.copy()
    contours, hierarchy = cv2.findContours(contour_finder, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw the contours, just for fun - this doesn't even look as pretty as rectangles would
    edge_map_color = cv2.cvtColor(edge_map_bw, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(edge_map_color, contours, -1, util.RED, 1)

    # OK here are rectangles, they're totally prettier
    musical_objects = []
    for contour in contours:
        musical_objects.append(cv2.boundingRect(contour))

    # error checking? merge disconnected components?

    for x1, y1, w, h in musical_objects:
        cv2.rectangle(edge_map_color, (x1, y1), (x1+w, y1+h), util.RED, 1)

    util.debug('found %d musical objects: %s' % (len(musical_objects), str(musical_objects)))

    """ Show the segmented image
    util.show('Segmented', edge_map_color)
    """

    util.show('Segmented', edge_map_color, True)



