import os
import cv2
import uuid
from src.canny_edge_detection_cv2 import *


# inputs the video array and outputs the contour of the head.
def largest_contour(img_array):
    #list contains the collection of contours that have suspected head contour
    head_contour_list = []

    #itterating through each image in img array
    for img in img_array:
        #runs head contour that gets the contour from the img
        contour, img_canny = head_contour(img)

        #asses the contour using its area
        area = cv2.contourArea(contour)



        #if the list is empty asign the first one
        if not head_contour_list:
            head_contour_list.append([contour, area, img])
            continue
        #if the area of the img is larger than the past frames replace it
        if area > head_contour_list[0][1]:
            head_contour_list.pop(0)
            head_contour_list.append([contour, area, img])

            #saving function to see the contour it outputs
            path = 'E:/test'
            contour_name = str(uuid.uuid4())
            cv2.drawContours(img, contour, -1, (0, 255, 0), 3)
            cv2.drawContours(img_canny, contour, -1, (0, 255, 0), 3)
            cv2.imwrite(os.path.join(path , contour_name + '.jpg'), img)
            cv2.imwrite(os.path.join(path , contour_name + 'canny.jpg'), img_canny)
        else:
            continue


    return head_contour_list

#outputs the contour of the head
def head_contour(img):
    img_canny = auto_canny(img)
    #uses sobel edge setector to generate outlines
    img_canny = make_sobel_face(img_canny)



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
