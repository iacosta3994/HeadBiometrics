
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

path = 'Video_Tests\A_test_Sanawar.mp4'
img_array = split_frames(path)
pixel_mm = video_to_pixel_mm(img_array)
contour, area, img = largest_contour(img_array)

cv2.imshow(img)
print(pixel_mm)


#video_to_pixel_mm('Video_Tests\A_test_alex.mp4')
#video_to_pixel_mm('Video_Tests\B_test_alex.mp4')
#video_to_pixel_mm('Video_Tests\A_test_Audrey.mp4')
#video_to_pixel_mm('Video_Tests\B_test_Audrey.mp4')
#video_to_pixel_mm('Video_Tests\A_test_BryanB.mp4')
#video_to_pixel_mm('Video_Tests\B_test_BryanB.mp4')
#video_to_pixel_mm('Video_Tests\A_test_BryanL.mp4')
#video_to_pixel_mm('Video_Tests\B_test_BryanL.mp4')
#video_to_pixel_mm('Video_Tests\A_test_Fanny.mp4')
#video_to_pixel_mm('Video_Tests\B_test_Fanny.mp4')
#video_to_pixel_mm('Video_Tests\A_test_Konstantin.mp4')
#video_to_pixel_mm('Video_Tests\B_test_Konstantin.mp4')
#video_to_pixel_mm('Video_Tests\A_test_Sanawar.mp4')
#video_to_pixel_mm('Video_Tests\B_test_Sanawar.mp4')
#video_to_pixel_mm('Video_Tests\A_test_Seb.mp4')
#video_to_pixel_mm('Video_Tests\B_test_Seb.mp4')
#video_to_pixel_mm('Video_Tests\A_test_Tom.mp4')
#video_to_pixel_mm('Video_Tests\B_test_Tom.mp4')
