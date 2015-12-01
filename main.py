from __future__ import print_function

import sys
import numpy
import cv2

import util
import persp


def main():
    if len(sys.argv) < 2:
        print('error: specify image filename', file=sys.stderr)
        return 1
    else:
        filename = sys.argv[1]

    img = cv2.imread(filename)
    util.show(filename, img)

    # find page & correct for perspective
    img2 = persp.find_page(img)

    util.show(filename + ' (corrected)', img2)

    cv2.waitKey(0)


if __name__ == '__main__':
    sys.exit(main())
