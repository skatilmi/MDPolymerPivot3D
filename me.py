import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from mpl_toolkits.mplot3d import Axes3D


class MonteCarlo:
    def __init__(self, seed=None, n=2):
        '''
        position as [x,y] while x,y are lists
        '''

        self.n = n
        if seed == None:
            random.seed(seed)
        else:
            self.seed = seed
        self.upperlimit = 0
        if self.n == 3:
            self.pos_trace = [[0, 0, 0]]
            self.directions = [
                [-1, 0, 0],
                [1, 0, 0],
                [0, -1, 0],
                [0, 1, 0],
                [0, 0, -1],
                [0, 0, 1]
            ]
        elif self.n == 2:
            self.pos_trace = [[0, 0]]
            self.directions = [
                [-1, 0],
                [1, 0],
                [0, -1],
                [0, 1],
            ]

    def arraying(self):
        self.pos_trace = np.array(self.pos_trace).T

    def stepOnLattice(self, n):
        if self.n == 3:
            for _ in range(n):
                x1, y1, z1 = random.choice(self.directions)
                x0, y0, z0 = self.pos_trace[-1]
                self.pos_trace.append([x0+x1, y0+y1, z0+z1])
        elif self.n == 2:
            for _ in range(n):
                x1, y1 = random.choice(self.directions)
                x0, y0 = self.pos_trace[-1]
                self.pos_trace.append([x0+x1, y0+y1])

    def stepOnLatticeSAW(self, moves):
        if self.n == 2:
            for _ in range(moves):
                x0, y0 = self.pos_trace[-1]
                c = 0
                for i in self.directions:
                    if [i[0]+x0, i[1]+y0] in self.pos_trace:
                        c += 1
                if c == 4:
                    self.upperlimit = _
                    break
                contains = True
                while contains:
                    x1, y1 = random.choice(self.directions)
                    x0, y0 = self.pos_trace[-1]
                    next_pos = [x0+x1, y0+y1]

                    if next_pos in self.pos_trace:
                        contains = True
                    else:
                        contains = False
                        self.pos_trace.append(next_pos)
        if self.n == 3:
            for _ in range(moves):
                x0, y0, z0 = self.pos_trace[-1]
                c = 0
                for i in self.directions:
                    if [i[0]+x0, i[1]+y0, i[2]+z0] in self.pos_trace:
                        c += 1
                if c == 6:
                    self.upperlimit = _
                    break
                contains = True
                while contains:
                    x1, y1, z1 = random.choice(self.directions)
                    x0, y0, z0 = self.pos_trace[-1]
                    next_pos = [x0+x1, y0+y1, z0+z1]

                    if next_pos in self.pos_trace:
                        contains = True
                    else:
                        contains = False
                        self.pos_trace.append(next_pos)
        self.start = self.pos_trace[0]
        self.end = self.pos_trace[-1]

    def pivot(self, tries=1):
        '''
        2d:
        step 1:
            randomly choose pivot somewhere on the walk
        step 2:
            choose one lattice symmetry on the pivot 6-1 possibilities at first
        step 3:
            try to apply the new symmetry
        repeat from step 1
        '''
        pass

    def plot(self):
        self.arraying()
        if self.n == 2:
            plt.plot(self.pos_trace[0], self.pos_trace[1])
            plt.plot(self.pos_trace[0], self.pos_trace[1], '.')
            plt.plot(*self.start, 'r.', markersize=13, label='start')
            plt.plot(*self.end, 'b.', markersize=13, label='end')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.legend()
        elif self.n == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(*self.pos_trace)
            ax.plot(*self.pos_trace)
            ax.scatter(*self.start, marker='^')
            ax.text(*self.start, 'start', color='red')
            ax.scatter(*self.end, marker='^')
            ax.text(*self.end, 'end', color='blue')
            plt.tight_layout()
        plt.show()


def sandbox(n):
    if False:
        current_max = 0
        current_maxSeed = 0
        for i in range(100000):
            random.seed(i)
            P = MonteCarlo(n=n, seed=i)
            P.stepOnLatticeSAW(1000)
            if P.upperlimit > current_max:
                current_max = P.upperlimit
                current_maxSeed = P.seed
                print(
                    f'biggest maze with n = {current_max} with seed {current_maxSeed}')

    # for 2D is seed = 6845 with 503 nodes
    # for 3D is seed = 327 with 986 nodes
    else:
        if n == 2:
            seed = 6845
        elif n == 3:
            seed = 327

        P = MonteCarlo(n=n, seed=seed)
        P.stepOnLatticeSAW(moves=1000)
        P.plot()
        print(list(P.pos_trace[0]))
        print(list(P.pos_trace[1]))
        print(list(P.pos_trace[2]))


sandbox(n=3)

P = MonteCarlo(n=3, seed=None)
P.stepOnLatticeSAW(moves=1000)
P.plot()


def Aufgabe():

    def getAbstand(p1, p2):
        k = 0
        for i, j in zip(p1, p2):
            k += (i-j)**2
        return np.sqrt(k)

    dicts = {}
    for j in range(10):
        #P = MonteCarlo(n=3, seed=327)
        P = MonteCarlo(n=3, seed=j)
        P.stepOnLatticeSAW(moves=1000)
        start = P.start

        number_of_nodes = []
        abstand_start_end = []

        for index, value in enumerate(P.pos_trace):
            number_of_nodes.append(index)
            abstand_start_end.append(getAbstand(start, value))

        # print(number_of_nodes)
        # print(abstand_start_end)
        #plt.plot(number_of_nodes, abstand_start_end, '.')

    k = [np.mean(l) for l in abstand_start_end]
    plt.plot(number_of_nodes, k, '.')

    plt.ylabel('<Ree^2>')
    plt.xlabel('N')
    plt.show()


# Aufgabe()
