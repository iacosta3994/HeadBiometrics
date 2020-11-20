import numpy as np
import cv2
import copy
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import widest_img
from src.keep_above_points import keep_img_above_points
from src.face_contour_width import img_head_contour_side
from src.distance_between_points import dis_in_points



def point_return(img):
    points = []

    def point_select(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(x, " ", y)
        elif event == cv2.EVENT_LBUTTONUP:
            points.append([x, y])
            print(x, " ", y)


    cv2.namedWindow('img')
    print("after named window")

    cv2.setMouseCallback('img', point_select)
    print("after setMouseCallback")

    cv2.imshow("img", img)
    print("imshow")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("break")
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return points[-2], points[-1]



def side_mm_metrics(path):
    img_array = split_frames(path)
    print("img_array: complete")

    side_head_img = widest_img(img_array)
    print('side_head_img: complete')
    if side_head_img is None:
        print("side_head_img is None")
    else:

        print("select length points")
        eyebrows, back = point_return(side_head_img)
        print("eyebrows, back: complete")
        length = dis_in_points( eyebrows,  back)
        print('length complete')


        print("select front to nape")
        front, nape = point_return(side_head_img)
        print('front and nape: complete')
        img_duplicate = side_head_img.copy()
        __, front2nape = img_head_contour_side(img_duplicate, front, nape, ret_contour = True)
        print('front2nape: complete')


        pixel_mm = video_to_pixel_mm(img_array)
        print("pixel_mm: complete")
        front2nape = int(front2nape * pixel_mm[0])
        print('final front2nape: complete')
        length = int(length * pixel_mm[0])
        print('final length: complete')
    return front2nape, length
