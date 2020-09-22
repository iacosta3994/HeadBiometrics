
import os
import sys
import numpy as np
import matplotlib
import cv2
from src.key_frame_extraction import *
from src.canny_edge_detection_cv2 import *
from src.mag_stripe_search import *
from src.video_to_mm import *

img_array = split_frames(path)
pixel_mm = video_to_pixel_mm(img_array)
print(pixel_mm)
