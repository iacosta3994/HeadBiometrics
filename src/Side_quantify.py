import numpy as np
import cv2


from src.face_contour_width import img_head_contour_side



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




def side_mm_metrics_postmtrp(wide_img, pixel_mm):


    def solve_length(img):
        print("select length points")
        eyebrows, back = point_return(img)
        length = dis_in_points( eyebrows,  back)
        return length


    length = solve_length(wide_img)
    length = int(length * pixel_mm[0])

    def solve_f2nape(img):
        print("select front to nape")
        front, nape = point_return(img)

        front2nape = img_head_contour_side(img, front, nape)
        return front2nape

    front2nape = solve_f2nape(wide_img)
    front2nape = int(front2nape * pixel_mm[0])
    front2nape = int(front2nape * .5)


    return front2nape, length
