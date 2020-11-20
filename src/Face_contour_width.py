import os
import cv2
import uuid
import numpy as np
from src.canny_edge_detection_cv2 import neural_edge_detection, auto_canny, make_sobel_face
from src.keep_above_points import keep_img_above_points


def get_contour(contours):
    if len(contours) == 2:
        contour = contours[0]
    elif len(contours) == 3:
        contour = contours[1]
    else:
        raise Exception(
            ("contour in face_contour_width is not working as expected,findContours changed output type check opencv documentation for updates"))
    return contour


def head_contour(img):

    img = auto_canny_face(img, sigma=.22)

    img = make_sobel_face(img)


    # creates list of contours
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # grabs the largest contour

    contour = get_contour(contours)
    if len(contour) == 0:
        return None , None

    contour = max(contour, key = cv2.contourArea)

    return contour, img

def img_head_contour_side(img, pointA, pointB, ret_contour = True):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    __, img_canny = neural_edge_detection(img, ret_contour=True)

    top_img = keep_img_above_points(img_canny, pointA, pointB)
    contour, img_canny = head_contour(top_img)
    contour_length = cv2.arcLength(contour, closed=False)
    contour_length = contour_length / 2

    if ret_contour == True:
        return  contour, contour_length
    else:
        return contour_length, img_canny

def img_head_contour_front(img, ret_contour = True):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    contour, img_canny = neural_edge_detection(img, ret_contour=True)

    contour_length = cv2.arcLength(contour, closed=False)

    if ret_contour == True:
        return  contour, contour_length
    else:
        return contour_length, img_canny
