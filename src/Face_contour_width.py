import os
import cv2
import uuid
from src.canny_edge_detection_cv2 import *

def segment_img(img):
    img_canny = make_canny_face(img)

    cnts = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    cv2.drawContours(img_canny, c, -1, (0, 255, 0), 3)

    path = 'E:/test'
    contour_name = str(uuid.uuid4())
    cv2.imwrite(os.path.join(path , 'contour_name'+'.jpg'), img_canny)
