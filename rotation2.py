import numpy as np
import matplotlib.pyplot as plt
import random


def rotate_around_pivot(xy, radians, pivot=[0, 0]):
    radians *= -1
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


MC_x = [0, 0, 1, 1, 0, 0, 1, 2, 3, 3, 4, 4, 4, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, -1, -1, -
        1, -2, -2, -3, -4, -5, -6, -7, -7, -6, -6, -5, -5, -4, -3, -3, -3, -2, -1, 0, 0, -1, -2]
MC_y = [0, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 3, 4, 5, 6, 6, 5, 5, 6, 6, 5, 5, 6, 6,
        7, 8, 8, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9, 8, 8, 8, 9, 10, 10, 10, 10, 9, 9, 9]


class PivotAlg:
    def __init__(self, walk):
        self.walk = walk
        self.walk_traces = [self.walk]
        self.pivot_point_traces = [np.array([0, 0])]
        self.walklen = len(self.walk)
        self.rotation_traces = [0]

    def tryrotate(self):
        t = random.choice([1, 2, 3])
        radians = np.pi / 2 * t
        pivot = random.randint(0, self.walklen)

        pivotPoint = self.walk_traces[-1][pivot]
        Sub1_MC = self.walk_traces[-1][:pivot]
        Sub2_MC = self.walk_traces[-1][pivot:]  # enthält den pivot point

        # rotate randomly one subgroup:
        Sub2_MC_prime = np.array(rotate_around_pivot(
            Sub2_MC.T, radians=radians, pivot=pivotPoint)).T

        c = 0
        for i in Sub1_MC:
            for j in Sub2_MC_prime:
                if np.all(i - j == np.zeros(2)):
                    print(f'rotation failed: {i}')
                    return
#                    c += 1
#                    break
#                break
#            break
        if c == 0:
            self.walk_traces.append(np.append(Sub1_MC, Sub2_MC_prime, axis=0))
            self.pivot_point_traces.append(pivotPoint)
            self.rotation_traces.append(t)

    def forceN(self, n):
        while len(self.walk_traces) < n+1:
            self.tryrotate()

    def getBounds(self):
        xmin, xmax, ymin, ymax = 0, 0, 0, 0

        for i in self.walk_traces:
            xmin_ = np.min(i.T[0])
            xmax_ = np.max(i.T[0])
            ymin_ = np.min(i.T[1])
            ymax_ = np.max(i.T[1])
            if xmin > xmin_:
                xmin = xmin_
            if ymin > ymin_:
                ymin = ymin_
            if xmax < xmax_:
                xmax = xmax_
            if ymax < ymax_:
                ymax = ymax_
        return xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5

    def plot(self):
        xmin, xmax, ymin, ymax = self.getBounds()

        bounds = np.array([[xmin, ymin], [xmax, ymin], [
                          xmax, ymax], [xmin, ymax], [xmin, ymin]]).T
        for i in range(1, len(self.walk_traces)):
            plt.figure()
            plt.title(
                f'#rotation iteration: {i}, rotation: {self.rotation_traces[i]}[pi/2]')
            plt.plot(*self.walk_traces[i-1].T, color='silver')
            plt.plot(*self.walk_traces[i-1].T, '.',
                     markersize=12, color='dimgray')

            plt.plot(*bounds, 'k')
            plt.plot(*self.walk_traces[i].T, linewidth=2)
            plt.plot(*self.walk_traces[i].T, '.', markersize=6)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.plot(*self.pivot_point_traces[i], 'r2', markersize=12)


MC_prev = np.array([MC_x, MC_y]).T
P = PivotAlg(MC_prev)
P.forceN(4)
P.plot()
plt.show()


#n = 10
#bounds = np.array([[-n, -n], [n, -n], [n, n], [-n, n], [-n, -n]]).T
# plt.show()
