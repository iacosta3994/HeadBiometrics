
import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import *
from src.canny_edge_detection_cv2 import *
from src.mag_stripe_search import *
from src.video_to_mm import *

pixel_mm = video_to_pixel_mm(('Video_Tests\B_test_Konstantin.mp4'), ('./data/B_frame'), 1)
print(pixel_mm)
