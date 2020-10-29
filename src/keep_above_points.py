import cv2
import numpy as np


def keep_img_above_points(image, pointA, pointB):

    pA1, pA2 = pointA
    pB1, pB2 = pointB
    # slope of line

    m = float(pB2 - pA2) / float(pB1 - pA1)
    c = pB2 - m*pB1

    if image is None:
        print("image is None_ cant continue 'keep img above points.py'")

    (imgy, imgx, imgc) = image.shape
    mask = np.zeros((imgy, imgx, imgc), np.uint8)

    for y in np.arange(0, pB2 + 1):
        for x in np.arange(0, pA1+1):

            if y > m*x + c:

                mask[y,x] = [255, 255, 255]


    final_img = cv2.merge((image[:, :, 0], image[:, :, 1], image[:, :, 2], mask[:, :, 0]))
    while True:
        cv2.imshow('final_img', final_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    return final_img
