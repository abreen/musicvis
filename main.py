from __future__ import print_function

import sys
import numpy
import cv2

import params
import util
import persp
import notes


def main():
    if len(sys.argv) < 3:
        print('%s: error: invalid arguments' % sys.argv[0], file=sys.stderr)
        print('usage: %s PATH_TO_IMAGE PATH_TO_PARAMS' % sys.argv[0], file=sys.stderr)
        return 1
    else:
        filename = sys.argv[1]
        params_path = sys.argv[2]

    img = cv2.imread(filename)
    util.show(filename, img)

    params.load(params_path)

    # find page & correct for perspective
    img2 = persp.find_page(img)

    if img2 is None:
        return 2

    util.show(filename + ' (corrected)', img2)

    staves = notes.find_staff(img2)

    cv2.waitKey(0)


if __name__ == '__main__':
    sys.exit(main())
