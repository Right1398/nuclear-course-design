import numpy as np
from scipy import integrate
import time
from multiprocessing import pool
import multiprocessing
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import math
import os
import threading
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

#length of side / cm
a_constant = 18

#diffusion coeffcient
D_constant = 1.77764

#macroscopic capture cross section / cm-1
Sigma_f = 0.0104869

#neutron yield
nu = 2.5

#macroscopic react cross section / cm-1
Sigma_r = 0.0142676

#Parameter
x_constant = 1.0
c_constant = 10
h_constant = 0.0000001
init_lamda = 1.0


#points number

INNER_POINTS_NUMBER = (a_constant - 1)**3
SURFACE_POINTS_NUMBER = 6 * (a_constant - 1)**2
OUTSIDE_POINTS_NUMBER = 6 * (a_constant - 1)**2
EXPECT_OUTSIDE_POINTS_NUMBER = INNER_POINTS_NUMBER + SURFACE_POINTS_NUMBER
POINTS_NUMBER = INNER_POINTS_NUMBER + SURFACE_POINTS_NUMBER + OUTSIDE_POINTS_NUMBER

FILE_NAME = ''

class Points():
    inner_points = list()
    outside_points = list()

    fore_points = list()
    back_points = list()
    left_points = list()
    right_points = list()
    top_points = list()
    bottom_points = list()

    expect_outside_points = list()
    points = list()

    expect_outside_points_list = list()
    expect_outside_points_array = np.zeros((1,1))

    outside_points_list = list()
    outside_points_array = np.zeros((1,1))

    points_list = list()
    points_array = np.zeros((1,1,1))

    def update(self):
        self.expect_outside_points = (self.inner_points + self.left_points  + self.right_points+ self.fore_points + self.back_points + self.top_points + self.bottom_points)
        self.points = self.expect_outside_points + self.outside_points

        self.expect_outside_points_list = [self.expect_outside_points[i] for i in range(EXPECT_OUTSIDE_POINTS_NUMBER)]
        self.expect_outside_points_array = np.array(self.expect_outside_points_list)

        self.outside_points_list = [self.outside_points[i]  for i in range(OUTSIDE_POINTS_NUMBER)]
        self.outside_points_array = np.array(self.outside_points_list)

        self.points_list = [self.points[i] for i in range(POINTS_NUMBER)]
        self.points_array = np.array(self.points_list)


class PointProcess():

    def __init__(self):
        pass

    def fix_point(self, innerPointNuM = 64, surfacePointNum = 96, outsidePointNum = 96):
        points = Points()

        for i in range(1, a_constant ):
            for j in range(1, a_constant ):
                for k in range(1, a_constant ):
                    points.inner_points.append( [i, j, k/a_constant*h_constant])


        for i in range(1, a_constant ):
            for j in range(1,a_constant ):
                points.fore_points.append([a_constant, i, j/a_constant*h_constant])
                points.back_points.append( [0, i, j/a_constant*h_constant])
                points.top_points.append([i, j, h_constant])
                points.bottom_points.append([i, j, 0])
                points.left_points.append([i,0, j/a_constant*h_constant])
                points.right_points.append([i, a_constant, j/a_constant*h_constant])

        for m in range(outsidePointNum):
            x = 0.0
            y = 0.0
            z = 0.0
            while(0 <= x <= a_constant and 0 <= y <= a_constant ):
                x = (np.random.rand() - np.random.rand()) * 5 * a_constant
                y = (np.random.rand() - np.random.rand()) * 5 * a_constant
                z = (np.random.rand() - np.random.rand()) * 5 * h_constant

            points.outside_points.append( [x, y, z])


        points.update()
        return(points)








    def setPoint(self, innerPointNum=4, surfacePointNum=6, outsidePointNum=6):
        points = Points()

        points.inner_points = [self.creatPoint('inner') for i in range(innerPointNum)]

        points.left_points = [[self.creatPoint('surface'), 0, self.creatPoint('surface')/a_constant*h_constant] for i in
                              range(round(surfacePointNum / 6))]  # (x, 0, z)
        points.fore_points = [[a_constant, self.creatPoint('surface'), self.creatPoint('surface')/a_constant*h_constant] for i in
                              range(round(surfacePointNum / 6))]  # (a, y, z)
        points.right_points = [[self.creatPoint('surface'), a_constant, self.creatPoint('surface')/a_constant*h_constant] for i in
                               range(round(surfacePointNum / 6))]  # (x, a, z)
        points.back_points = [[0, self.creatPoint('surface'), self.creatPoint('surface')/a_constant*h_constant] for i in
                            range(round(surfacePointNum / 6))]  # (0, y, z)
        points.top_points = [[self.creatPoint("surface"), self.creatPoint("surface"), h_constant] for i in
                             range(round(surfacePointNum  / 6))]
        points.bottom_points = [[self.creatPoint("surface"), self.creatPoint("surface"), 0] for i in
                                range(surfacePointNum - 5 * round(surfacePointNum / 6))]

        points.outside_points = [self.creatPoint('outside') for i in range(outsidePointNum)]

        points.update()

        return (points)

    def creatPoint(self, pointType):
        x = 0.0
        y = 0.0
        z = 0.0
        if pointType == 'inner':

            x = np.random.rand() * a_constant
            y = np.random.rand() * a_constant
            z = np.random.rand() * h_constant


            return([x, y, z])
        if pointType == 'surface':
            return(np.random.rand() * a_constant)
        if pointType == 'outside':
            while(0 <= x <= a_constant and 0 <= y <= a_constant and 0 <= z <= h_constant):
                x = np.random.rand() * 10 * a_constant - 5 * a_constant
                y = np.random.rand() * 10 * a_constant - 5 * a_constant
                z = np.random.rand() * 10 * h_constant - 5 * h_constant
            return([x, y, z])

class matrixProcess():
    points = Points()
    KAA = np.zeros((POINTS_NUMBER, POINTS_NUMBER))
    coefficient_matrix = np.zeros((POINTS_NUMBER, POINTS_NUMBER))
    inv_coefficient_matrix = np.zeros((POINTS_NUMBER, POINTS_NUMBER))
    Psi_matrix = np.zeros((EXPECT_OUTSIDE_POINTS_NUMBER, POINTS_NUMBER))

    def __init__(self, points, c = 1.0):
        self.points = points
        self.c = c

    def CalculateKAA(self):
        KAA = np.zeros((POINTS_NUMBER, POINTS_NUMBER))
        c = self.c
        NI =INNER_POINTS_NUMBER
        NE = len(self.points.expect_outside_points)
        NL = len(self.points.left_points)
        NF = len(self.points.fore_points)
        NR = len(self.points.right_points)
        NB = len(self.points.back_points)
        NT = len(self.points.top_points)
        NBO = len(self.points.bottom_points)
        for i in range(POINTS_NUMBER):
            for j in range(POINTS_NUMBER):
                if i < NE:
                    xi = self.points.points[i][0]
                    yi = self.points.points[i][1]
                    zi = self.points.points[i][2]
                    xj = self.points.points[j][0]
                    yj = self.points.points[j][1]
                    zj = self.points.points[j][2]
                    KAA[i][j] = - D_constant*math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)**(-6 / c**2 + 4 * ((xi-xj)**2+(yi - yj)**2+(zi - zj)**2) / c**4) + Sigma_r * math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)
                else:
                    xi = self.points.points[i-NE+NI][0]
                    yi = self.points.points[i-NE+NI][1]
                    zi = self.points.points[i-NE+NI][2]
                    xj = self.points.points[j][0]
                    yj = self.points.points[j][1]
                    zj = self.points.points[j][2]
                    if i < (NE + NL ):#左右
                        KAA[i][j] =  math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)#((yi - yj) / (((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 + c ** 2) ** 0.5))
                    elif i < (NE + NL + NR):
                        KAA[i][j] =  math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)*(-2 / c**2 + 4 * (yi - yj)**2 / c**4)
                    elif i < (NE + NL + NR + NF):#前后
                        KAA[i][j] = math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)*(-2 / c**2 + 4 * (xi - xj)**2 / c**4)#((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 + c ** 2) ** 0.5
                    elif i < (NE + NL + NR + NF + NB):

                        KAA[i][j] =  math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)

                    elif i < (NE + NL + NR + NF + NB + NT):#上下
                        KAA[i][j] = ((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 + c ** 2) ** 0.5
                    elif i < (NE + NL + NR + NF + NB + NT + NBO):
                        KAA[i][j] = ((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 + c ** 2) ** 0.5
        self.KAA = KAA
        return(KAA)

    def CalculatePsimatrix(self):
        Psi_matrix = np.zeros((EXPECT_OUTSIDE_POINTS_NUMBER, POINTS_NUMBER))
        c = self.c
        for i in range(EXPECT_OUTSIDE_POINTS_NUMBER):
            for j in range(POINTS_NUMBER):
                xi = self.points.points[i][0]
                yi = self.points.points[i][1]
                zi = self.points.points[i][2]
                xj = self.points.points[j][0]
                yj = self.points.points[j][1]
                zj = self.points.points[j][2]
                Psi_matrix[i][j] = math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)
        self.Psi_matrix = Psi_matrix
        return (Psi_matrix)

    def CalculateAll(self):
        self.CalculateKAA()
        self.coefficient_matrix = self.KAA
        self.inv_coefficient_matrix = np.linalg.inv(self.coefficient_matrix)
        self.CalculatePsimatrix()
        return(self)

def Solve(c, a, xi, yi, zi, i, integrate_matrix_list):
    f = lambda z, y, x:( math.exp(-1 *((xi - x) ** 2 + (yi - y) ** 2 + (zi - z) ** 2 )/c**2))
    result = integrate.tplquad(f, 0, a, 0, a, 0, a/a_constant*h_constant)
    integrate_matrix_list[i] = [result[0]]

def IntegrateCoefficientMatrix(initialPoints):
    c = c_constant
    a = a_constant

    integrate_matrix_list = multiprocessing.Manager().list(range(POINTS_NUMBER))
    p = pool.Pool()
    for i in range(POINTS_NUMBER):
        xi = initialPoints.points[i][0]
        yi = initialPoints.points[i][1]
        zi = initialPoints.points[i][2]

    p.apply_async(Solve, args=(c, a, xi, yi, zi, i, integrate_matrix_list))
    p.close()
    p.join()
    integrate_matrix = np.array(integrate_matrix_list).T
    return (integrate_matrix)

def IntegrateCoefficientMatrix(initialPoints):
    c = c_constant
    a = a_constant

    integrate_matrix_list = multiprocessing.Manager().list(range(POINTS_NUMBER))
    p = pool.Pool()
    for i in range(POINTS_NUMBER):
        xi = initialPoints.points[i][0]
        yi = initialPoints.points[i][1]
        zi = initialPoints.points[i][2]
        p.apply_async(Solve, args=(c, a, xi, yi, zi, i, integrate_matrix_list))
    p.close()
    p.join()
    integrate_matrix = np.array(integrate_matrix_list).T
    return (integrate_matrix)

class DrawAndData():
    points = Points()
    x_values = np.zeros((1,1))
    y_values = np.zeros((1,1))
    z_values = np.zeros((1,1))
    Z = 0.0

    x_mean = np.zeros((1,1))
    y_mean = np.zeros((1,1))

    x_var = np.zeros((1,1))
    y_var = np.zeros((1,1))

    file_name = ''

    def __init__(self, points):
        self.points = points
        self.x_values = self.points.points_array[:,0]
        self.y_values = self.points.points_array[:,1]
        self.z_values = self.points.points_array[:,2]

        self.x_mean = np.mean(self.x_values)
        self.y_mean = np.mean(self.y_values)

        self.x_var = np.var(self.x_values)
        self.y_var = np.var(self.y_values)

        self.file_name = 'data/a_%d_inner_%d_outside_%d_c_%d/'%(a_constant, INNER_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER, c_constant)

    def CalculateZ(self, a_matrix):
        X = np.arange(0, 50, 1)
        Y = np.arange(0, 50, 1)
        X, Y = np.meshgrid(X, Y)

        z_val = np.array(a_matrix)[:,0]
        for i in range (0,POINTS_NUMBER):
            self.Z = self.Z + z_val[i] * np.exp(-1 *((X - self.x_values[i]) ** 2 + (Y - self.y_values[i]) ** 2 )/c_constant**2)
                     #((X - self.x_values[i]) ** 2 + (Y - self.y_values[i]) ** 2 + c_constant**2) ** 0.5
            #math.exp(-1 *((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2 )/c**2)*(-2 / c**2 + 4 * (yi - yj)**2 / c**4)
    '''      
    def Random3D(self):
        fig = plt.figure()
        ax = Axes3D(fig)
        X = np.range(-10,60,1)
        Y = np.range(-10,60,1)
        X, Y = np.meshgrid(X, Y)
    '''
    def PlotRandomPoint(self):
    
        plt.scatter(self.x_values,self.y_values)
        plt.xlim((-10,60))
        plt.ylim((-10,60))
        file_name = FILE_NAME + '/Random.png'
        plt.savefig(file_name)

        plt.close()

    def PlotNeutronFlux(self, a_matrix):
        fig = plt.figure()
        ax = Axes3D(fig)

        X = np.arange(0, 50, 1)
        Y = np.arange(0, 50, 1)
        X, Y = np.meshgrid(X, Y)
        z_val = np.array(a_matrix)[:,0]

        ax.plot_surface(X, Y, self.Z)
        plt.savefig("%s.png"%np.random.rand())
        plt.show()

    
    def PlotHeat(self):
        plt.imshow(self.Z,interpolation = 'nearest' , cmap = 'jet')
        plt.colorbar()
        file_name = FILE_NAME + '/Heat.png'
        plt.savefig(file_name)

        plt.close()
    '''
    def FileOutput(self, lamda, iterationFlag):
        file_name = self.file_name + 'data.csv'

        with open(file_name,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = ["x:",self.x_var,self.x_mean,"y:",self.y_var,self.y_mean,"lamda:",lamda,'iteration flag',iterationFlag]
        
            csv_write.writerow(data_row)
    '''
    '''
    def LamdaOutput(self, lamda):
        
        file_name = FILE_NAME + '/lamda.csv'
        
        
        with open(file_name,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = [lamda]
        
            csv_write.writerow(data_row)
        
        
        np.savetxt(file_name, lamda,delimiter=',')

    '''



def main():
    # initial_time = time.time()
    initial_points = PointProcess().fix_point(INNER_POINTS_NUMBER, SURFACE_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER)
    '''
    draw = DrawAndData(initial_points)
    draw.PlotRandomPoint()
    '''
    # print('PointProcess %f'%(time.time() - initial_time))
    # time1 = time.time()
    draw = DrawAndData(initial_points)
    matrix = matrixProcess(initial_points, c_constant).CalculateAll()
    # print('MatrixProcess %f'%(time.time() - time1))
    f_matrix_before = np.vstack((np.ones((EXPECT_OUTSIDE_POINTS_NUMBER, 1)), np.zeros((OUTSIDE_POINTS_NUMBER, 1))))

    x = x_constant
    lamda = init_lamda
    lamda_before = init_lamda
    # time2 = time.time()
    integrate_coefficient_matrix = IntegrateCoefficientMatrix(initial_points)

    #lamda_array = np.array([lamda])

    # print('solve Process %f'%(time.time() - time2))
    a_matrix_before = np.ones((POINTS_NUMBER, 1))

    iteration_flag = False

    for i in range(1000000):
        a_matrix = x / lamda * np.dot(matrix.inv_coefficient_matrix, f_matrix_before)  # A matrix

        f_matrix_no_zeros = nu * Sigma_f * np.dot(matrix.Psi_matrix, a_matrix)
        f_matrix = np.vstack((f_matrix_no_zeros, np.zeros((OUTSIDE_POINTS_NUMBER, 1))))

        lamda = lamda_before * (np.dot(integrate_coefficient_matrix, a_matrix) / np.dot(integrate_coefficient_matrix,
                                                                                     a_matrix_before))

        f_matrix_before = f_matrix
        a_matrix_before = a_matrix
        lamda_before = lamda
        if abs(lamda_before - lamda)<0.0001:
            print(lamda)
            print("true")
            break

        draw.CalculateZ(a_matrix)
        draw.PlotNeutronFlux(a_matrix)
        '''
        if np.isnan(np.min(a_matrix)) or lamda == 0:
            # np.nan_to_num(a_matrix)
            lamda_array = np.append(lamda_array, [[np.NaN]])
            break

        lamda_array = np.append(lamda_array, lamda[0][0])
        '''

    draw.CalculateZ(a_matrix)
    draw.PlotNeutronFlux(a_matrix)
    #draw.LamdaOutput(lamda)
if __name__ == '__main__':
    main()
