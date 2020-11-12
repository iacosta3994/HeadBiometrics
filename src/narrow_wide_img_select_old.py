import cv2
import numpy as np
from src.face_contour_width import head_contour

left_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_righteye_2splits.xml')
eye_pair = cv2.CascadeClassifier(cv2.data.haarcascades + 'frontalEyes35x16.xml')

def narrowest_img(img_array):     # Inputs frame-filename to scan for the narrowest head img

    # establishing var outside loop
    dist = None
    ret_img = None

    # itteration for each img
    for img in img_array:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        left_eye = left_eye_cascade.detectMultiScale(img, 1.05, 4)
        right_eye = right_eye_cascade.detectMultiScale(img, 1.05, 4)
        both_eyes = eye_pair.detectMultiScale(img, 1.05, 4)

        # if both eyes found
        if (type(left_eye).__module__ == np.__name__) and (type(right_eye).__module__ == np.__name__):

            # itterating through both combos of left and right
            for (bex, bey, bew, beh)  in both_eyes:
                for (lex, ley, lew, leh) in left_eye:

                    for (rex, rey, rew, reh) in right_eye:


                        test_img = cv2.rectangle(img, (lex, ley), (lex + lew, ley +leh), (255,0,0), 2)
                        cv2.imshow('left eye', test_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        test_img = cv2.rectangle(test_img, (rex, rey), (rex + rew, rey +reh), (0,255,0), 2)
                        cv2.imshow('right eye', test_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        test_img = cv2.rectangle(test_img, (bex, bey), (bex + bew, bey +beh), (0,0,255), 2)
                        cv2.imshow('Both eyes with both eyes', test_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

                        # distance between points of detected eyes
                        img_dist = np.sqrt((rex-lex)**2 + (rey - ley)**2)

                        # if dist not established first img creates value
                        if dist == None:

                            dist = img_dist
                            ret_img = img
                        # when larger distance detected updates return img
                        if img_dist < dist:
                            dist = img_dist
                            ret_img = img




    if ret_img is not None:
        return ret_img




def widest_img(img_array, img_orientation):

    widest_head_img = None
    largest_contour = None
    # itterating through each image in img array
    for img in img_array:
        # runs head contour that gets the contour from the img
        for orientation in img_orientation:
            beginY, endY, beginX, endX = orientation
            og_img = img.copy()
            img = img[beginY:endY, beginX:endX]
            contour, img_canny = head_contour(img)

            contour_length = cv2.arcLength(contour, closed=False)
            if largest_contour == None or contour_length > largest_contour:
                cv2.drawContours(img, contour, -1, (0,255,0), 3)
                cv2.imshow('Widest img', img_canny)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                largest_contour = contour_length
                widest_head_img = og_img
            else:
                continue

    return widest_head_img


def front_head_select(img_array):
    modelFile = "src/tensor_pose_estimation/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "src/tensor_pose_estimation/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    for img in img_array:
        img = cv2.imread('test.jpg')
        h, w = img.shape[:2]
        og_img = img.copy()
        blob = cv2.dnn.blobFromImage(cv2.resize(og_img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
        net.setInput(blob)
        faces = net.forward()
    #to draw faces on image
        for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > 0.5:
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
