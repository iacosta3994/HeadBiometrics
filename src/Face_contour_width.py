import os
import cv2
import uuid
from src.canny_edge_detection_cv2 import *




def img_head_contour(img):
    if img is None:
        print("Can't open image file")
        return None
        
    contour, img_canny = head_contour(img)
    return contour, img_canny

# outputs the contour of the head
def head_contour(img):
    if img is None:
        print("Can't open image file")
        return None

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
