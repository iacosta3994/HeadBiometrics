
import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import *
from src.video_to_mm import *
from src.narrow_wide_img_select import narrowest_img
from src.path_finder import *
from src.face_detect_auto_crop import crop_above_eyes
from src.face_contour_width import img_head_contour


path = 'Video_Tests\A_test_alex.mp4'
img_array = split_frames(path)
pixel_mm, mag_xy = video_to_pixel_mm(img_array)

narrow_head_img = narrowest_img(img_array)
if narrow_head_img is None:
    print("Narrow head img is none")
else:
    cv2.imwrite("results.jpg",narrow_head_img)
    above_eyes = crop_above_eyes(narrow_head_img,mag_xy)
    if above_eyes is None:
        print("above eyes is none")
    else:
        front_contour = img_head_contour(above_eyes)



#contour_start = contour_bottom_left(front_contour)
#contour_end = contour_bottom_right(front_contour)

#contour_mask = make_mask_from_contour(front_contour)

#astar(contour_mask, contour_start, contour_end, allow_diagonal_movement=True)
