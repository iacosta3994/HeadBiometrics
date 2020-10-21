import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import widest_img
from src.xy_img_points_click import point_return
from src.keep_above_points import keep_img_above_points
from src.face_contour_width import img_head_contour
from src.distance_between_points import dis_in_points


def side_mm_metrics(path):
    img_array = split_frames(path)
    pixel_mm, mag_xy = video_to_pixel_mm(img_array)

    side_head_img = widest_img(img_array)
    if side_head_img is None:
        print("side_head_img is None")
    else:
        front, nape = point_return(side_head_img)
        final_img = keep_img_above_points(side_head_img.copy, front, nape)
        _, _, front2nape = img_head_contour(final_img)

        pointA, pointB = point_return(side_head_img)
        length = dis_in_points(pointA, pointB)

        front2nape = int(front2nape * pixel_mm[0])
        length = int(length * pixel_mm[0])
    return front2nape, length
