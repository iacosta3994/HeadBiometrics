import numpy as np
import cv2

from src.face_contour_width import ned_front
from src.contour_width_max import ret_contour_width



def front_mm_metrics_postmtrp(narrow_img, pixel_mm):

        main_contour, main_contour_length = ned_front(narrow_img)

        head_width_mm = int( ret_contour_width(main_contour)  * pixel_mm[0])

        ear_to_ear_mm = int(main_contour_length  * pixel_mm[0])

        # head width is subtracted because the original contour includes the ear to ear portion which shouldn't eb included.
        ear_to_ear_mm -= head_width_mm
        return ear_to_ear_mm, head_width_mm
