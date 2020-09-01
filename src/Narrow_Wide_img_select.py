''' this moduloe wil have 2 functions to find the narrowest object and the widest to determine which image to use to canny and pathfind. '''


import cv2
import numpy as np
from src.image_segmentation import *
from src.canny_edge_detection_cv2 import *

def narrowest_img (filename, ):     # Inputs frame-filename to scan for the narrowest head img
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml)
