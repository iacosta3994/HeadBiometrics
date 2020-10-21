import numpy as np

def dis_in_points(pointA, pointB):
    (xA, yA) = pointA
    (xB, yB) = pointB
    return round(sqrt((xA-xB)**2 + (yA - yB)**2))
