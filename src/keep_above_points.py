import cv2
import argparse
import numpy as np
from src.canny_edge_detection_cv2 import neural_edge_detection


def keep_img_above_points(img, pointA, pointB):

    x1, y1 = pointA
    x2, y2 = pointB
    # slope of line

    m = float(y2 - y1) / float(x2 - x1)
    c = y2 - m*x2

    if img is None:
        print("img is None_ cant continue 'keep img above points.py'")

    (imgy, imgx, imgc) = img.shape

    img_canny = neural_edge_detection(img, ret_contour = False)

    if len(img.shape) == 2:
        img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)


    for x in np.arange(0, imgx):
        for y in np.arange(min(y1,y2), imgy):

            if y > m*x + c:

                img_canny[y][x] = [0, 0, 0]

    cv2.imshow('keep_above_points', img_canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    return img_canny
