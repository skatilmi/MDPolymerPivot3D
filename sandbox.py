import numpy as np
from numpy import cos, sin, pi


def Rx(theta):
    t = theta
    return np.round(np.array([
        [1, 0, 0],
        [0, cos(t), -sin(t)],
        [0, sin(t), cos(t)],
    ]))


def Ry(theta):
    t = theta
    return np.round(np.array([
        [cos(t), 0, sin(t)],
        [0, 1, 0],
        [-sin(t), 0, cos(t)],
    ]))


# def Rz(point, pivot, radians):
def Rz(radians):
    t = radians
#    x1, x2, x3 = point
#    p1, p2, p3 = pivot
#    if quarterCircles == 1:
#        return -(x2-p2)+p1, (x1-p1)+p2, x3
#    if quarterCircles == 1:
#        return -(x2-p2)+p1, (x1-p1)+p2, x3

    return np.round(np.array([
        [cos(t), -sin(t), 0],
        [sin(t), cos(t), 0],
        [0, 0, 1],
    ]))


print(Rx(pi))
print(Ry(pi))
print(Rz(pi))
