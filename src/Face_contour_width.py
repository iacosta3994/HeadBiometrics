import os
import cv2
import uuid
from src.canny_edge_detection_cv2 import *



def largest_contour(img_array):
    largest_contour = []
    for img in img_array:
        contour, img_canny = head_contour(img)
        area = cv2.contourArea(contour)
        if not largest_contour:
            largest_contour.append([contour, area, img])
            continue
        if area > largest_contour[0][1]:
            largest_contour.pop(0)
            largest_contour.append([contour, area, img])

            cv2.drawContours(img_canny, contour, -1, (0, 255, 0), 3)
            path = 'E:/test'
            contour_name = str(uuid.uuid4())
            cv2.imwrite(os.path.join(path , contour_name + '.jpg'), img_canny)
        else:
            continue


    return largest_contour

def head_contour(img):
    img_canny = make_canny_face(img)

    contour = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = get_contour(contour)
    contour = max(contour, key=cv2.contourArea)

    return contour, img_canny

def get_contour(contour):
    if len(contour) == 2:
        contour = contour[0]
    elif len(contour) == 3:
        contour = contour[1]
    else:
        raise Exception(("contour in face_contour_width is not working as expected,findContours changed output type check opencv documentation for updates"))
    return contour
