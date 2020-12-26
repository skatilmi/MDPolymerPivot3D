from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import random
import glob
import sys
from mpl_toolkits.mplot3d import Axes3D


# these are the rotation operations at x,y,z axis in either pi/2, pi or pi/2*3

def Rx(point, pivot, quarterCircles=1):
    x1, x2, x3 = point
    p1, p2, p3 = pivot
    if quarterCircles == 1:
        return x1, -(x3-p3)+p2, (x2-p2)+p3
    if quarterCircles == 2:
        return x1, (x3-p3)+p2, -(x2-p2)+p3
    if quarterCircles == 3:
        return x1, -x2+2*p2, -x3+2*p3


def Ry(point, pivot, quarterCircles=1):
    x1, x2, x3 = point
    p1, p2, p3 = pivot
    if quarterCircles == 1:
        return (x3-p3)+p1, x2, -(x1-p1)+p3
    if quarterCircles == 2:
        return -(x3-p3)+p1, x2, (x1-p1)+p3
    if quarterCircles == 3:
        return -x1+2*p1, x2, -x3+2*p3


def Rz(point, pivot, quarterCircles=1):
    x1, x2, x3 = point
    p1, p2, p3 = pivot
    if quarterCircles == 1:
        return -(x2-p2)+p1, (x1-p1)+p2, x3
    if quarterCircles == 2:
        return (x2-p2)+p1, -(x1-p1)+p2, x3
    if quarterCircles == 3:
        return -x1+2*p1, -x2+2*p2, x3


class PivotAlg:
    def __init__(self, walk):
        self.walk = walk
        self.walk_traces = [self.walk]
        self.pivot_point_traces = [np.array([0, 0, 0])]
        self.walklen = len(self.walk)
        self.rotation_traces = [0]
        self.choices = [Rx, Ry, Rz]
        self.end2end_distance = []

    def tryrotate(self):
        # choose a random pivot point in (start, end) of the whole current walk:
        pivot = random.randint(1, self.walklen-2)
        pivotPoint = self.walk_traces[-1][pivot]

        # split the walk at the choosen pivot. Due symmetry, it does not depend which subpolymer we continue with
        Sub1_MC = self.walk_traces[-1][:pivot]
        Sub2_MC = self.walk_traces[-1][pivot:]

        # now we determine, which axis are allowed to be rotated at
        # meaning that we dont rotate at the axis parallel to the direction, the pivot anchor is pointing to
        anchor = Sub1_MC[-1] - pivotPoint
        choicesConfiguration = [i for i, j in zip(
            self.choices, anchor) if j == 0]

        axisRotation = random.choice(choicesConfiguration)

        # choose, if we rotate counterclockwise or clockwise
        quarterCircles = random.choice([1, 2, 3])

        Sub2_MC_prime = np.array(axisRotation(
            Sub2_MC.T, pivot=pivotPoint, quarterCircles=quarterCircles)).T

        c = 0
        for i in Sub1_MC:
            for j in Sub2_MC_prime:
                if np.all(i - j == np.zeros(3)):
                    #print(f'rotation failed: {i}')
                    print('failed')
                    return

        if c == 0:
            self.walk_traces.append(np.append(Sub1_MC, Sub2_MC_prime, axis=0))
            self.pivot_point_traces.append(pivotPoint)
            # self.rotation_traces.append(direction)
            d = self.walk_traces[-1][0]-self.walk_traces[-1][-1]
            self.end2end_distance.append(np.sqrt(d[0]**2+d[1]**2+d[2]**2))
            print(
                f'number of different configurations: {len(self.walk_traces)-1} and polymer length: {self.walklen}')

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
        #zbounds = [zmax+0.2, zmin-0.2, zmax+0.2, xmin - 0.2, xmax+0.2, xmin-0.2, xmax+0.2, xmin-0.2]
        zbounds = [zmax+0.2, zmin-0.2, zmax+0.2, zmin -
                   0.2, zmax+0.2, zmin-0.2, zmax+0.2, zmin-0.2]

        return xbounds, ybounds, zbounds

    def forceN(self, n):
        while len(self.walk_traces) < n+1:
            self.tryrotate()

    def savedistances2file(self, path='distances'):
        np.savetxt(path, self.end2end_distance, delimiter=',')

    def plot(self):
        bounds = self.getBounds()
        for i in range(0, len(self.walk_traces)):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(*self.walk_traces[i].T)
            ax.scatter(*self.walk_traces[i].T, color='orange')
            ax.scatter(*bounds)
            ax.scatter(*self.pivot_point_traces[i], s=70, color='k')

            plt.tight_layout()
        plt.show()


def rotateMazes():
    paths = [i.replace('\\', '/') for i in glob.glob('MAZES/len*/*')]
    for path in paths:
        maze_len = int(path.split('/')[1].split('len')[-1])
        name = path.split('/')[-1].split('.')[0]
        if maze_len > 1001:
            MC_prev = np.genfromtxt(path, unpack=True).T
            P = PivotAlg(MC_prev)
            P.forceN(50)
            P.savedistances2file(path=f'distances/distances_{name}_{maze_len}')

            data = np.genfromtxt(f'distances/distances_{name}_{maze_len}')
            with open('distances.txt', 'a') as a:
                a.write(f'{name},{np.mean(data)},{np.std(data)}\n')

    todo = glob.glob('distances/*')
    with open('distances.txt', 'a') as a:
        for i in todo:
            data = np.genfromtxt(i)
            name = i.split('_')[-1]
            a.write(f'{name},{np.mean(data)},{np.std(data)}\n')


def Aufgabe():
    def lin(x, a):
        return a*x

    N, M, S = np.genfromtxt(
        'distances.txt', delimiter=',', unpack=True, skip_header=1)
    x = np.log(N)
    y = np.log(M)*2
    s = np.log(S)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    xi = np.linspace(min(x), max(x), 51)

    popt, pcov = curve_fit(lin, x, y, sigma=S)
    perr = np.sqrt(np.diag(pcov))

    ax1.errorbar(x=x, y=y, yerr=s, xerr=0, fmt='.')
    ax1.plot(xi, lin(xi, *popt),
             label=f'nu = {round(popt[0]/2,3)} +- {round(perr[0]/2,3)}')
    ax2.plot(x, lin(x, *popt)-y, '.')

    ax1.set_ylabel('ln(<Ree^2>)')
    ax2.set_ylabel('residuals')
    ax2.set_xlabel('ln(N)')

    print(f'nu = {popt[0]/2} +- {perr[0]/2}')

    ax1.grid()
    ax2.grid()
    ax1.legend()
    plt.tight_layout()
    plt.show()


# rotateMazes()

Aufgabe()
