import cv2
import numpy as np
from side_quantify import  side_head_img

def point_select(event, x, y, flags, params):

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, ' ', y)

if __name__ == "__main__":

    cv2.imshow("img", side_head_img)
    cv2.setMouseCallback("img", point_select)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
