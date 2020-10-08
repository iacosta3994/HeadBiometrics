
import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import *
from src.canny_edge_detection_cv2 import *
from src.mag_stripe_search import *
from src.video_to_mm import *
from src.face_contour_width import *
from src.narrow_wide_img_select import narrowest_img
from src.path_finder import *
from src.face_detect_auto_crop import crop_above_eyes


path = 'Video_Tests\A_test_self.mp4'
img_array = split_frames(path)
pixel_mm, mag_xy = video_to_pixel_mm(img_array)
narrow_head_img = narrowest_img(img_array)
above_eyes = crop_above_eyes(narrow_head_img,mag_xy)
if above_eyes is None:
    print("Can't open above eyes file")

front_contour = img_head_contour(above_eyes)



#contour_start = contour_bottom_left(front_contour)
#contour_end = contour_bottom_right(front_contour)

#contour_mask = make_mask_from_contour(front_contour)

#astar(contour_mask, contour_start, contour_end, allow_diagonal_movement=True)

print(pixel_mm)


contour_name = str(uuid.uuid4())
path = 'E:/test/'
cv2.drawContours(above_eyes, contour, -1, (0, 255, 0), 3)
cv2.imwrite(os.path.join(path ,  'result.jpg'), above_eyes)
