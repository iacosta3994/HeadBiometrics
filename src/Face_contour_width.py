import os
import cv2
import uuid
import numpy as np

from src.canny_edge_detection_cv2 import neural_edge_detection, auto_canny_face, make_sobel_face
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

def head_contour_side(img):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (3,3), 0)
    img = auto_canny_face(img, sigma= 0.55)
    img = make_sobel_face(img)
    # creates list of contours
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # grabs the largest contour

    contour = get_contour(contours)
    if len(contour) == 0:
        return None , None

    contour = max(contour, key = cv2.contourArea)
    return contour

def img_head_contour_side_SMPL(img, pointA, pointB):


    f2nape = keep_img_above_points(img, pointA, pointB)
    contour = head_contour_side(f2nape)

    cv2.drawContours(f2nape, [contour],-1, (0,255,0), 3)
    cv2.imshow("img_head_contour_side_post_NED", f2nape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    f2nape_length = cv2.arcLength(contour, closed = True)
    #might need to divide by 2
    return f2nape_length




def img_head_contour_side(img, pointA, pointB):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    print ("turned bgr")
    ned_img = neural_edge_detection(img, ret_contour= False)
    print("ned")
    top_img = keep_img_above_points(ned_img, pointA, pointB)
    print("keep_above_points")
    contour = head_contour_post_NED(top_img)
    print("contours")

    cv2.drawContours(top_img, [contour],-1, (0,255,0), 3)
    cv2.imshow("img_head_contour_side_post_NED", top_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    if len(img_canny.shape) == 2:
        img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(img_canny, contour,-1, (0,255,0), 3)
    cv2.imshow("img_head_contour_front", img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #contour_length = cv2.arcLength(contour, closed=True)
    contour_length = len(contour)

    #contour_length = contour_length / 2

    if ret_contour == True:
        return  contour, contour_length
    else:
        return contour_length, img_canny
