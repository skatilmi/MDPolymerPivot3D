import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from mpl_toolkits.mplot3d import Axes3D
MC_x = [0, -1, -1, -1, -2, -2, -3, -3, -3, -3, -3, -3, -3, -3, -3, -2, -2, -2, -2, -3, -3, -2, -2, -2, -2, -3, -4, -4, -3, -3, -2, -2, -2, -1, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 3, 3, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 7, 6, 6, 6, 6, 5, 5, 4, 4, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 7, 6, 6, 6, 6, 5, 4, 3, 3, 2, 2, 2, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 2, 3, 3, 2, 2, 1, 1, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 7, 8, 9, 9, 8, 8, 9, 10, 10, 10, 10, 9, 9, 9, 9, 9,
        8, 8, 9, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 11, 11, 10, 10, 10, 11, 11, 11, 12, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 12, 12, 12, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 11, 11, 11, 10, 10, 9, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 8, 8, 8, 8, 7, 7, 7, 8, 9, 9, 10, 10, 10,
        10, 11, 11, 10, 10, 11, 11, 12, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 11, 11, 11, 11, 11, 10, 10, 9, 9, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 16, 16, 16, 15, 15, 16, 17, 17, 17, 17, 16, 16, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 14, 14, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 12, 12, 11, 11, 11, 11, 11, 11, 11, 12, 12]
MC_y = [0, 0, -1, -2, -2, -3, -3, -2, -1, -1, -2, -3, -3, -3, -2, -2, -1, -1, 0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 4, 4, 4, 5, 5, 5, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 9, 9, 9, 9, 9, 8, 8, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 11, 11, 12,
        12, 12, 12, 12, 11, 10, 9, 9, 8, 8, 7, 7, 6, 5, 5, 5, 6, 6, 6, 6, 6, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 6, 6, 6, 6, 6, 5, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 5, 6, 6, 5, 5, 5,
        5, 5, 6, 6, 5, 4, 4, 3, 3, 3, 3, 3, 3, 2, 1, 1, 0, 0, 1, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2, 2, 1, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -2, -3, -4, -4, -4, -4, -4, -4, -3, -3, -3, -2, -2, -2, -2, -2, -2, -2, -1, -1, -1, -1, -2, -2, -3, -3, -4, -4, -4, -4, -3, -2, -1, -1, 0, 0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 5, 5, 5, 5, 4, 4, 4, 3, 3, 2, 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 6, 6, 6, 6, 6, 5, 4, 4, 4, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 5, 5, 5, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 6, 6, 6, 5, 5, 6, 6, 5, 4, 4, 4, 4, 4]
MC_z = [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -2, -3, -3, -3, -3, -2, -2, -2, -3, -3, -3, -4, -4, -4, -4, -4, -4, -4, -4, -3, -3, -3, -3, -4, -4, -5, -5, -5, -5, -4, -4, -4, -5, -6, -6, -6, -6, -6, -6, -7, -7, -7, -7, -6, -6, -6, -5, -5, -4, -3, -3, -3, -3, -2, -1, -1, -1, -2, -2, -2, -2, -3, -3, -4, -4, -4, -4, -4, -3, -2, -2, -3, -3, -3, -2, -2, -1, -1, -1, -2, -2, -3, -3, -4, -4, -5, -5, -5, -5, -5, -6, -6, -7, -8, -8, -8, -8, -8, -8, -8, -7, -7, -6, -6, -6, -6, -6, -6, -6, -5, -5, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -6, -5, -4, -4, -4, -4, -3, -2, -2, -2, -2, -2, -2, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 1, 1, 1, 1, 1, 2, 2, 3, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 4, 4, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
        5, 5, 4, 4, 5, 6, 6, 7, 8, 9, 9, 9, 8, 7, 7, 6, 6, 6, 7, 7, 6, 5, 5, 5, 5, 5, 4, 4, 3, 2, 2, 2, 2, 1, 1, 2, 3, 4, 5, 5, 6, 6, 7, 7, 6, 6, 7, 7, 8, 8, 7, 6, 6, 6, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8, 7, 7, 7,
        8, 9, 9, 9, 9, 9, 8, 7, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6, 6, 6, 6, 7, 8, 8, 8, 7, 7, 7, 8, 8, 7, 7, 7, 7, 7, 7, 8, 8, 8, 9, 10, 10, 9, 9, 10, 10, 10, 11, 12, 12, 11, 11, 11, 11, 11, 11, 10, 10, 10, 9, 9, 8, 7, 7, 6, 5, 5, 5, 6, 6, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 9, 10, 10, 10, 11, 11, 10, 9, 9, 9, 9, 9, 10, 10, 9, 8, 8, 7, 7, 6, 6, 7, 8, 9, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 10, 10, 10, 9, 8,
        8, 9]

#MC_x = [0, 1, 1, 1]
#MC_y = [0, 0, 0, -1]
#MC_z = [0, 0, 1, 1]


def rotate_around_pivot(xyz, direction, pivot=[0, 0, 0]):
    """Rotate a point around a given point.

    I call this the "high performance" version since we're caching some
    values that are needed >1 time. It's less readable than the previous
    function but it's faster.
    """
    x, y = xy
    offset_x, offset_y = pivot
    adjusted_x = x - offset_x
    adjusted_y = y - offset_y

    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)

    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y

    return np.round([qx, qy])


def rotate_along_pivot():
    pass


class PivotAlg:
    def __init__(self, walk):
        self.walk = walk
        self.walk_traces = [self.walk]
        self.pivot_point_traces = [np.array([0, 0, 0])]
        self.walklen = len(self.walk)
        self.rotation_traces = [0]

    def tryrotate(self):
        direction = random.choice([1, 2, 3, 4])
        pivot = random.randint(0, self.walklen)

        pivotPoint = self.walk_traces[-1][pivot]
        Sub1_MC = self.walk_traces[-1][:pivot]
        Sub2_MC = self.walk_traces[-1][pivot:]  # enthält den pivot point

        # rotate randomly one subgroup:
        Sub2_MC_prime = np.array(rotate_around_pivot(
            Sub2_MC.T, direction=direction, pivot=pivotPoint)).T

        c = 0
        for i in Sub1_MC:
            for j in Sub2_MC_prime:
                if np.all(i - j == np.zeros(3)):
                    print(f'rotation failed: {i}')
                    return

        if c == 0:
            self.walk_traces.append(np.append(Sub1_MC, Sub2_MC_prime, axis=0))
            self.pivot_point_traces.append(pivotPoint)
            self.rotation_traces.append(direction)

    def getBounds(self):
        xmin, xmax, ymin, ymax, zmin, zmax = 0, 0, 0, 0, 0, 0

        for i in self.walk_traces:
            xmin_ = np.min(i.T[0])
            xmax_ = np.max(i.T[0])
            ymin_ = np.min(i.T[1])
            ymax_ = np.max(i.T[1])
            zmin_ = np.min(i.T[2])
            zmax_ = np.max(i.T[2])

            if xmin > xmin_:
                xmin = xmin_
            if xmax < xmax_:
                xmax = xmax_

            if ymin > ymin_:
                ymin = ymin_
            if ymax < ymax_:
                ymax = ymax_

            if zmin > zmin_:
                zmin = zmin_
            if zmax < zmax_:
                zmax = zmax_
        n = 1
        xbounds = [xmax+0.2, xmax+0.2, xmax+0.2, xmax +
                   0.2, xmin-0.2, xmin-0.2, xmin-0.2, xmin-0.2]
        ybounds = [ymax+0.2, ymax+0.2, ymin-0.2, ymin -
                   0.2, ymax+0.2, ymax+0.2, ymin-0.2, ymin-0.2]
        zbounds = [zmax+0.2, zmin-0.2, zmax+0.2, xmin -
                   0.2, xmax+0.2, xmin-0.2, xmax+0.2, xmin-0.2]

        return xbounds, ybounds, zbounds

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(*self.walk.T)
        ax.scatter(*self.walk.T, color='orange')

        plt.tight_layout()
        ax.scatter(*self.getBounds())

        plt.show()


MC_prev = np.array([MC_x, MC_y, MC_z]).T
P = PivotAlg(MC_prev)
P.plot()
