import os
import cv2
import uuid
import numpy as np
from src.canny_edge_detection_cv2 import *


def img_head_contour(img_samples_array):
    main_canny = None
    main_contour = None
    main_contour_length = None

    if img_samples_array is None:
        print("Can't open sample array")
        return main_canny, main_contour, main_contour_length

    for img in img_samples_array:

        img_duplicate = img.copy()

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        contour, img_canny = head_contour(img)

        if contour is None:
            continue

        cv2.drawContours(img_duplicate, contour, -1, (0,255,0), 3)
        cv2.imshow('Contours', img_duplicate)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        contour_length = cv2.arcLength(contour, closed=False)

        if contour_length is None:
            print("contour length is none on face contour width")
            continue

        if main_contour_length is None or contour_length > main_contour_length:
            main_canny = img_canny
            main_contour = contour
            main_contour_length = contour_length
        else:
            continue

    if main_canny is None and main_contour is None and main_contour_length is None:
        print("no return values found")
    return main_canny, main_contour, main_contour_length

# outputs the contour of the head


def head_contour(img):

    img = auto_canny_face(img, sigma=.22)

    img = make_sobel_face(img)


    # creates list of contours
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # grabs the largest contour

    contour = get_contour(contours)
    if len(contour) == 0:
        return None , None

    contour = max(contour, key = cv2.contourArea)

    return contour, img


def get_contour(contours):
    if len(contours) == 2:
        contour = contours[0]
    elif len(contours) == 3:
        contour = contours[1]
    else:
        raise Exception(
            ("contour in face_contour_width is not working as expected,findContours changed output type check opencv documentation for updates"))
    return contour
