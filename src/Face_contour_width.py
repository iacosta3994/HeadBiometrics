import os
import cv2
import uuid
import numpy as np
from src.canny_edge_detection_cv2 import *




def img_head_contour(img_samples_array):
    if img_samples_array is None:
        print("Can't open sample array")
        return None

    main_canny = None
    main_contour = None
    main_contour_length = None

    for img in img_samples_array:
        contour, img_canny = head_contour(img)

        if contour is None:
            continue

        contour_length = cv2.arcLength(contour, closed =False)

        if contour_length is None:
            print("contour length is none on face contour width")
            continue

        if main_contour_length is None:
            main_canny = img_canny
            main_contour = contour
            main_contour_length = contour_length

        elif contour_length > main_contour_length:
            main_canny = img_canny
            main_contour = contour
            main_contour_length = contour_length
        else:
            continue


    if main_canny is None and main_contour is None and main_contour_length is None:
        print("no return values found")
    return  main_canny, main_contour, main_contour_length

# outputs the contour of the head
def head_contour(img):

    img = auto_canny_face(img, sigma=.22)

    img = make_sobel_face(img)

    # creates list of contours
    contour = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # grabs the largest contour
    contour = get_contour(contour)
    # in the case there is 2 contours made sure to grab the one with largest area
    contour = max(contour, key=cv2.contourArea)

    return contour, img


def get_contour(contour):
    if len(contour) == 2:
        contour = contour[0]
    elif len(contour) == 3:
        contour = contour[1]
    else:
        raise Exception(
            ("contour in face_contour_width is not working as expected,findContours changed output type check opencv documentation for updates"))
    return contour
