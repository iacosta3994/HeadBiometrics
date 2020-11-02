import cv2
import numpy as np
from src.canny_edge_detection_cv2 import auto_canny_face


def keep_img_above_points(image, pointA, pointB):

    x1, y1 = pointA
    x2, y2 = pointB
    # slope of line

    m = float(y2 - y1) / float(x2 - x1)
    c = y2 - m*x2

    if image is None:
        print("image is None_ cant continue 'keep img above points.py'")

    (imgy, imgx, imgc) = image.shape
    #mask = np.zeros((imgy, imgx, imgc), np.uint8)
    img_canny = auto_canny_face(image)
    img_canny = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    for x in np.arange(x1, imgx):
        for y in np.arange(0, imgy):
            if y < m*x + c:

                img_canny[y][x] = [0, 0, 0]



    '''while True:
        cv2.imshow('final_img', img_canny)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    cv2.imwrite('mask.jpg', mask)
    cv2.imwrite('img_canny.jpg', img_canny)
    cv2.imwrite('image.jpg', image)'''
    return img_canny
