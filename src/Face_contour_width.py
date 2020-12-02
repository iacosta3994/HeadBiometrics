import os
import cv2
import uuid
import numpy as np
import argparse
from src.canny_edge_detection_cv2 import neural_edge_detection, auto_canny_face, make_sobel_face



def get_contour(contours):
    if len(contours) == 2:
        contour = contours[0]
    elif len(contours) == 3:
        contour = contours[1]
    else:
        raise Exception(
            ("contour in face_contour_width is not working as expected,findContours changed output type check opencv documentation for updates"))
    return contour

def ned_front(img, ret_img = False):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    contour, ned_img = neural_edge_detection(img, ret_contour=True)

    if len(ned_img.shape) == 2:
        ned_img = cv2.cvtColor(ned_img, cv2.COLOR_GRAY2BGR)

    contour_length = len(contour)

    if ret_img == True:
        return contour_length, ned_img
    else:
        return contour, contour_length
'''
# uses neural edge detection to locate edge of head
def img_head_contour_side_NED(img, pointA, pointB):


    def keep_img_above_points(img, pointA, pointB):

        x1, y1 = pointA
        x2, y2 = pointB
        # slope of line

        m = float(y2 - y1) / float(x2 - x1)
        c = y2 - m*x2

        if img is None:
            print("img is None_ cant continue 'keep img above points.py'")

        (imgy, imgx, imgc) = img.shape

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        contour, ned_img = neural_edge_detection(img, ret_contour=True)

        if len(ned_img.shape) == 2:
            ned_img = cv2.cvtColor(ned_img, cv2.COLOR_GRAY2BGR)


        for x in np.arange(0, imgx):
            for y in np.arange(min(y1,y2), imgy):

                if y > m*x + c:
                    ned_img[y][x] = [0, 0, 0]


        return ned_img

    f2nape = keep_img_above_points(img, pointA, pointB)

    def head_contour_side(f2nape):

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        # creates list of contours
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # grabs the largest contour

        contour = get_contour(contours)
        if len(contour) == 0:
            return None

        contour = max(contour, key = cv2.contourArea)
        return contour

    contour = head_contour_side(f2nape)

    cv2.drawContours(f2nape, [contour],-1, (0,255,0), 3)
    cv2.imshow("img_head_contour_side", f2nape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    f2nape_length = cv2.arcLength(contour, closed = True)

    return f2nape_length
'''


#locates front to nape with cv tools
def img_head_contour_side(img, pointA, pointB):

    def keep_img_above_points(img, pointA, pointB):

        x1, y1 = pointA
        x2, y2 = pointB

        # slope of line
        m = float(y2 - y1) / float(x2 - x1)
        c = y2 - m*x2

        if img is None:
            print("img is None_ cant continue 'keep img above points.py'")

        (imgy, imgx, imgc) = img.shape

        img = auto_canny_face(img)
        img = make_sobel_face(img)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for x in np.arange(0, imgx):
            for y in np.arange(min(y1,y2), imgy):

                if y > m*x + c:
                    img[y][x] = [0, 0, 0]
        return img

    def head_contour_side(img):

        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # creates list of contours
        contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # grabs the largest contour

        contour = get_contour(contours)
        if len(contour) == 0:
            return None

        contour = max(contour, key = cv2.contourArea)
        return contour

    f2nape = keep_img_above_points(img, pointA, pointB)

    contour = head_contour_side(f2nape)

    f2nape_length = cv2.arcLength(contour, closed = True)

    f2nape_length -= 500


    return f2nape_length
