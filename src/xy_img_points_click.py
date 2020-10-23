import cv2
import numpy as np


def point_select(event, x, y, flags, params):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse_X, mouse_Y = x, y


def point_return(img):

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', point_select)

    while True:
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return mouse_X, mouse_Y
