import cv2
import numpy as np


def keep_img_above_points(image, pointA, pointB):

    # slope of line
    m = float(pointA[1] - pointB[1]) / float(pointA[0] - pointB[0])
    c = pointB[1] - m*pointB[0]

    y, x, _ = image.shape
    mask = np.zeros((y, x), np.uint8)

    for x in np.arange(0, pointA[0]+1):
        for y in np.arange(0, pointB[1] + 1):
            if y < m*x + c:
                mask[y][x] = (255, 255, 255)
    final_img = cv2.merge((image[:, :, 0], img[:, :, 1], img[:, :, 2], mask[:, :, 0]))

    return final_img
