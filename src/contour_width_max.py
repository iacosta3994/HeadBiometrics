import cv2
import numpy as np


def ret_contour_width(contour):
    box = cv2.minAreaRect(contour)
    box = cv2.boxPoints(box)
    box = np.int0(box)
    (tl, tr, br, bl) = box
    left_max = tl - bl
    right_max = tr - br
    if left_max[0] >= right_max[0]:
        (x, y) = left_max
        if x > y:
            return x
        else:
            return y
    else:
        (x, y) = right_max
        if x > y:
            return x
        else:
            return y
