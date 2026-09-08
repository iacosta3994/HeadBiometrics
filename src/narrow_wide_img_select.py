import sys
import cv2
import math
import numpy as np

from src.Proctoring_AI.face_detector import get_face_detector, find_faces
from src.Proctoring_AI.face_landmarks import get_landmark_model, detect_marks
from src.Proctoring_AI.head_pose_estimation import head_pose_points

from src.face_detect_auto_crop import crop_above_eyes

#performs head pose estimation and returns both narrow and wide
def narrow_wide_img(img_array):
    face_model = get_face_detector()
    landmark_model = get_landmark_model()

    size = img_array[0].shape

    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

    narrow_img = None
    narrow_ang1 = None
    narrow_ang2 = None
    narrow_img_eyes_xy = None

    wide_img = None
    wide_ang1 = None
    wide_ang2 = None


    for img in img_array:

        faces = find_faces(img, face_model)
        for face in faces:
            marks = detect_marks(img, landmark_model, face)

            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)



            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)


            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(img, rotation_vector, translation_vector, camera_matrix)
        try:
            m = (p2[1] - p1[1])/(p2[0] - p1[0])
            ang1 = int(math.degrees(math.atan(m)))
        except:
            continue

        try:
            m = (x2[1] - x1[1])/(x2[0] - x1[0])
            ang2 = int(math.degrees(math.atan(-1/m)))
        except:
            continue



        if narrow_ang1 is None or abs(ang1) < narrow_ang1:
            if narrow_ang2 is None or abs(ang2) < narrow_ang2:
                narrow_ang1 = ang1
                narrow_ang2 = ang2
                narrow_img = img

                narrow_img_eyes_xy = (image_points[3])

                (xA, yA) = (image_points[3])
                (xB, yB) = (image_points[2])
                offset = (round(np.sqrt((xA-xB)**2 + (yA - yB)**2))/4)

                narrow_img_eyes_xy[1] -= offset

        if wide_ang2 is None or abs(ang2) > wide_ang2:

            wide_ang1 = abs(ang1)
            wide_ang2 = abs(ang2)
            wide_img = img

    narrow_img = crop_above_eyes(narrow_img, narrow_img_eyes_xy)

    return narrow_img, wide_img
