
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
print(pixel_mm, mag_xy)
narrow_head_img = narrowest_img(img_array)
if narrow_head_img is None:
    print("Narrow head img is none")
above_eyes = crop_above_eyes(narrow_head_img,mag_xy)
if above_eyes is None:
    print("above eyes is none")

front_contour = img_head_contour(above_eyes)



#contour_start = contour_bottom_left(front_contour)
#contour_end = contour_bottom_right(front_contour)

#contour_mask = make_mask_from_contour(front_contour)

#astar(contour_mask, contour_start, contour_end, allow_diagonal_movement=True)


contour_name = str(uuid.uuid4())
path = 'E:/test/'
cv2.drawContours(above_eyes, front_contour, -1, (0, 255, 0), 3)
cv2.imwrite(os.path.join(path ,  'result.jpg'), above_eyes)
