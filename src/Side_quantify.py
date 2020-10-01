
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
from src.path_finder import *

path = 'Video_Tests\B_test_BryanL.mp4'
img_array = split_frames(path)
pixel_mm = video_to_pixel_mm(img_array)
narrow_head_img = narrowest_img(img_array)
front_contour = img_head_contour(narrow_head_img)



contour_start = contour_bottom_left(front_contour)
contour_end = contour_bottom_right(front_contour)

contour_mask = make_mask_from_contour(front_contour)

astar(contour_mask, contour_start, contour_end)

print(pixel_mm)

path = 'E:/test'
contour_name = str(uuid.uuid4())
cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
cv2.imwrite(os.path.join(path ,  'result.jpg'), img)
