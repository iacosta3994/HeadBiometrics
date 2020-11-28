import numpy as np
import cv2
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import narrowest_img
from src.face_contour_width import img_head_contour_front
from src.contour_width_max import ret_contour_width


def front_mm_metrics(path):

    img_array = split_frames(path)

    pixel_mm = video_to_pixel_mm(img_array)

    narrow_head_img = narrowest_img(img_array)

    main_contour, main_contour_length = img_head_contour_front(narrow_head_img)

    ear_to_ear_mm = int(main_contour_length * pixel_mm[0])

    head_width_mm = int(ret_contour_width(main_contour) * pixel_mm[0])

    return ear_to_ear_mm, head_width_mm

def front_mm_metrics_postmtrp(img_array, pixel_mm):

        narrow_head_img = narrowest_img(img_array)

        main_contour, main_contour_length = img_head_contour_front(narrow_head_img)

        head_width_mm = int( ret_contour_width(main_contour)  * pixel_mm[0])

        ear_to_ear_mm = int(main_contour_length  * pixel_mm[0])

        # head width is subtracted because the original contour includes the ear to ear portion which shouldn't eb included.
        ear_to_ear_mm -= head_width_mm
        return ear_to_ear_mm, head_width_mm
