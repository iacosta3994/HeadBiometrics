
import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import narrowest_img
from src.face_detect_auto_crop import crop_above_eyes
from src.face_contour_width import img_head_contour
from src.contour_width_max import ret_contour_width

def front_mm_metrics(path):
    img_array = split_frames(path)
    pixel_mm, mag_xy = video_to_pixel_mm(img_array)

    narrow_head_img = narrowest_img(img_array)
    if narrow_head_img is None:
        print("Narrow head img is none")
    else:
        img_samples_array = crop_above_eyes(narrow_head_img,mag_xy)
        if img_samples_array is None:
            print("img_samples_array is none")
        else:
            main_canny, main_contour, main_contour_length = img_head_contour(img_samples_array)

            ear_to_ear_mm = int(main_contour_length * pixel_mm[0])

            head_width = ret_contour_width(main_contour)

            #cv2.imwrite("results.jpg",narrow_head_img)
            #cv2.imwrite("canny_image.jpg", main_canny)
    return ear_to_ear_mm , int(head_width)
