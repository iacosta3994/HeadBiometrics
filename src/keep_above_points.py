import cv2
import argparse
import numpy as np
from src.canny_edge_detection_cv2 import auto_canny_face, make_sobel_face, neural_edge_detection


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
