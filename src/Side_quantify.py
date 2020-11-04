import os
import sys
import numpy as np
import matplotlib
import cv2
import copy
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import widest_img
#from src.xy_img_points_click import point_return
from src.keep_above_points import keep_img_above_points
from src.face_contour_width import img_head_contour
from src.distance_between_points import dis_in_points

points = []
def point_select(event, x, y, flags, params):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, " ", y)
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        print(x, " ", y)
        points.append((x, y))

def point_return(img):
    global points


    cv2.namedWindow('img')
    cv2.setMouseCallback('img', point_select)


    while True:
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return points[-2], points[-1]


def side_mm_metrics(path):
    img_array = split_frames(path)
    pixel_mm, mag_xy = video_to_pixel_mm(img_array)

    side_head_img = widest_img(img_array)
    if side_head_img is None:
        print("side_head_img is None")
    else:
        print("select front to nape")
        front, nape = point_return(side_head_img)

        img_duplicate = side_head_img.copy()

        final_img = keep_img_above_points(img_duplicate,  front, nape)
        final_img = [final_img]
        _, _, front2nape = img_head_contour(final_img)

        print("select length points")
        eyebrows, back = point_return(side_head_img)

        length = dis_in_points( eyebrows,  back)

        front2nape = int(front2nape * pixel_mm[0])
        length = int(length * pixel_mm[0])
    return front2nape, length
