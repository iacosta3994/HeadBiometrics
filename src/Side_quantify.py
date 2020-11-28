import numpy as np
import cv2
import copy
from src.key_frame_extraction import split_frames
from src.video_to_mm import video_to_pixel_mm
from src.narrow_wide_img_select import widest_img
from src.keep_above_points import keep_img_above_points
from src.face_contour_width import img_head_contour_side_SMPL



def point_return(img):
    points = []

    def point_select(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        if event == cv2.EVENT_LBUTTONUP:
            points.append([x, y])


    cv2.namedWindow('img')

    cv2.setMouseCallback('img', point_select)


    cv2.imshow("img", img)


    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):

            cv2.destroyAllWindows()
            cv2.waitKey(0)
            return points[-2], points[-1]

def dis_in_points(pointA, pointB):
    (xA, yA) = pointA
    (xB, yB) = pointB

    return round(np.sqrt((xA-xB)**2 + (yA - yB)**2))


def side_mm_metrics(path):
    img_array = split_frames(path)


    side_head_img = widest_img(img_array)

    if side_head_img is None:
        print('Error: can not use image')
    else:

        def solve_length(img):
            print("select length")
            eyebrows, back = point_return(img)

            length = dis_in_points( eyebrows,  back)

            return length

        side_head_img_length = side_head_img.copy()
        length = solve_length(side_head_img_length)


        def solve_f2nape(img):
            print("select front to nape")
            front, nape = point_return(img)
            front2nape = img_head_contour_side_SMPL(img, front, nape)
            return front2nape

        side_head_img_front2nape = side_head_img.copy()
        front2nape = solve_f2nape(side_head_img_front2nape)

        pixel_mm = video_to_pixel_mm(img_array)

        front2nape = int(front2nape * pixel_mm[0])

        angle_correction = np.sqrt(pixel_mm[0]/2)
        length = int(length * angle_correction)

    return front2nape, length


def side_mm_metrics_postmtrp(img_array, pixel_mm):
    side_head_img = widest_img(img_array)
    if side_head_img is None:
        print("side_head_img is None")
    else:

        def solve_length(img):
            print("select length points")
            eyebrows, back = point_return(img)
            length = dis_in_points( eyebrows,  back)
            return length
        side_head_img_length = side_head_img.copy()
        length = solve_length(side_head_img_length)


        def solve_f2nape(img):
            print("select front to nape")
            front, nape = point_return(img)
            front2nape = img_head_contour_side_SMPL(img, front, nape)
            return front2nape

        side_head_img_front2nape = side_head_img.copy()
        front2nape = solve_f2nape(side_head_img_front2nape)
        front2nape = int(front2nape * pixel_mm[0])
        front2nape = int(front2nape * .5)
        length = int(length * pixel_mm[0])

    return front2nape, length
