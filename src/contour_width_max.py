import cv2
import numpy as np


def ret_contour_width(contour):
    # Builds a box around the inputed contour
    box = cv2.minAreaRect(contour)
    # collects data from box
    box = cv2.boxPoints(box)
    # changes format to 64 bit integer
    box = np.int0(box)
    # top Left ... bottom Left
    (tl, tr, br, bl) = box

    left_max = abs(tl - bl)
    right_max = abs(tr - br)
    top_max = abs(tl - tr)
    bottom_max = abs(bl - br)

    max_list = []
    max_list.append(max(left_max))
    max_list.append(max(right_max))
    max_list.append(max(top_max))
    max_list.append(max(bottom_max))

    return max(max_list)
