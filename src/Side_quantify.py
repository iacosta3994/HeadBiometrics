import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.keep_above_points import keep_img_above_points
from src.distance_between_points import dif_in_points

def side_mm_metrics(path):
    img_array = split_frames(path)
    pixel_mm, mag_xy = video_to_pixel_mm(img_array)

    side_head_img = widest_img(img_array)
    if side_head_img is None:
        print("side_head_img is None")
    else:
        pointA, pointB =
        final_img = keep_img_above_points(side_head_img, pointA, pointB)
        front2nape = img_mask_contour(final_img)
        length = dif_in_points(pointA, pointB)


    return front2nape, length
