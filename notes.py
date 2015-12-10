import numpy
import cv2

import params
import util

NOTE_DURATIONS = ['quarter', 'eighth', 'half', 'whole']


def find_notes(img, objs, staff_lines):
    # from 5 staff line coordinates, build coordinates of nine positions
    # (5 lines and four spaces between each line)

    staff_lines = sorted(staff_lines)

    avg = sum([abs(staff_lines[i] - staff_lines[i + 1]) \
              for i in range(len(staff_lines) - 1)]) / (len(staff_lines) - 1)
    avg2 = float(avg) / 2

    positions = {
        staff_lines[0] - avg2:       'G5',
        staff_lines[0]:              'F5',
        util.avg(staff_lines, 0, 1): 'E5',
        staff_lines[1]:              'D5',
        util.avg(staff_lines, 1, 2): 'C5',
        staff_lines[2]:              'B4',
        util.avg(staff_lines, 2, 3): 'A4',
        staff_lines[3]:              'G4',
        util.avg(staff_lines, 3, 4): 'F4',
        staff_lines[4]:              'E4',
        staff_lines[4] + avg2:       'D4',
        staff_lines[4] + 2 * avg2:   'C4',
        staff_lines[4] + 3 * avg2:   'B3',
    }

    notes = []
    img2 = cv2.GaussianBlur(img, (29, 29), 0)

    # sort by column (ascending) to add notes left-to-right
    objs = sorted(objs, key=lambda x: x[0])

    for c, r, w, h in objs:
        print(c, r, w, h)

        circles = cv2.HoughCircles(
                img2[r:r+h, c:c+w],
                cv2.cv.CV_HOUGH_GRADIENT,
                1,
                10,
                param1=50,
                param2=16
        )

        if circles is not None and len(circles[0]) == 1:
            x, y, radius = circles[0][0]

            dist, pos = min([( abs(r + y - p) , positions[p]) for p in positions.keys()])
            notes.append((pos, 'quarter'))

            # TODO determine note duration?

    # TODO sort notes by ascending x-position (for correct order)
    return notes
