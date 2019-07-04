import numpy as np
from scipy import integrate
import time
from multiprocessing import pool
import multiprocessing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import os
import threading

import matplotlib.cm as cm
from matplotlib.colors import LogNorm

#length of side / cm
a_constant = 50

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
c_constant = 1.0
init_lamda = 1.0


#points number

INNER_POINTS_NUMBER = 400
SURFACE_POINTS_NUMBER = 60
OUTSIDE_POINTS_NUMBER = 60
EXPECT_OUTSIDE_POINTS_NUMBER = INNER_POINTS_NUMBER + SURFACE_POINTS_NUMBER
POINTS_NUMBER = INNER_POINTS_NUMBER + SURFACE_POINTS_NUMBER + OUTSIDE_POINTS_NUMBER

FILE_NAME = ''

class Point():
    x = 0.0
    y = 0.0

    def __init__(self, x = 0.0, y = 0.0):
        self.x = x
        self.y = y
    
    def __str__(self):
        return '{:.4f},{:.4f}'.format(self.x, self.y)
    
    def x(self):
        return self.x

    def y(self):
        return self.y
    
    def value(self):
        return([self.x,self.y])

class Points():
    inner_points = list()
    outside_points = list()

    up_points = list()
    down_points = list()
    left_points = list()
    right_points = list()

    expect_outside_points = list()
    points = list()

    expect_outside_points_list = list()
    expect_outside_points_array = np.zeros((1,1))

    outside_points_list = list()
    outside_points_array = np.zeros((1,1))

    points_list = list()
    points_array = np.zeros((1,1))

    def __init__(self):
        pass

    def update(self):
        self.expect_outside_points = (self.inner_points + self.left_points + self.down_points + self.right_points + self.up_points)
        self.points = self.expect_outside_points + self.outside_points

        self.expect_outside_points_list = [self.expect_outside_points[i].value() for i in range(EXPECT_OUTSIDE_POINTS_NUMBER)]
        self.expect_outside_points_array = np.array(self.expect_outside_points_list) 

        self.outside_points_list = [self.outside_points[i].value() for i in range(OUTSIDE_POINTS_NUMBER)]
        self.outside_points_array = np.array(self.outside_points_list)

        self.points_list = [self.points[i].value() for i in range(POINTS_NUMBER)]
        self.points_array = np.array(self.points_list)

class PointProcess(Point):

    def __inif__(self):
        pass

    def setPoint(self, innerPointNum = 4, surfacePointNum = 4, outsidePointNum = 4):

        points = Points()

        points.inner_points = [self.creatPoint('inner') for i in range(innerPointNum)]

        points.left_points = [(Point(0, self.creatPoint('surface'))) for i in range(round(surfacePointNum / 4))]
        points.down_points = [(Point(self.creatPoint('surface'), 0)) for i in range(round(surfacePointNum / 4))]
        points.right_points = [Point(a_constant, self.creatPoint('surface')) for i in range(round(surfacePointNum / 4))]
        points.up_points = [(Point(self.creatPoint('surface'), a_constant)) for i in range(surfacePointNum - 3 * round(surfacePointNum / 4))]

        points.outside_points = [self.creatPoint('outside') for i in range(outsidePointNum)]
                

        points.update()

        return(points)


    def creatPoint(self, pointType):
        ''' 
        to ensure every side has least a point , parameter 'serface' only
        return a number from 0 to a.
        '''

        if pointType == 'inner':
            x = np.random.rand() * a_constant
            y = np.random.rand() * a_constant
            return(Point(x, y))
        
        if pointType == 'surface':
            return(np.random.rand() * a_constant)

        if pointType == 'outside':
            x = np.random.rand() * 10 * a_constant - 5 * a_constant
            if 0 <= x <= a_constant:
                x -= a_constant
            y = np.random.rand() * 10 * a_constant - 5 * a_constant
            if 0 <= y <= a_constant:
                y -= a_constant
            return(Point(x, y))

class matrixProcess(Point):
    points = Points()
    c = 1.0

    KDD = np.zeros((EXPECT_OUTSIDE_POINTS_NUMBER, EXPECT_OUTSIDE_POINTS_NUMBER))
    KDE = np.zeros((EXPECT_OUTSIDE_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER))
    KBD = np.zeros((SURFACE_POINTS_NUMBER, EXPECT_OUTSIDE_POINTS_NUMBER))
    KBE = np.zeros((SURFACE_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER))
    Psi_matrix = np.zeros((EXPECT_OUTSIDE_POINTS_NUMBER,POINTS_NUMBER))
    coefficient_matrix = np.zeros((POINTS_NUMBER, POINTS_NUMBER))
    inv_coefficient_matrix = np.zeros((POINTS_NUMBER, POINTS_NUMBER))

    def __init__(self, points, c = 1.0):
        self.points = points
        self.c = c
    
    def CalculateKDD(self):

        c = self.c


        matrix_square = np.sum(self.points.expect_outside_points_array ** 2, axis=1).reshape(EXPECT_OUTSIDE_POINTS_NUMBER,1)
        matrix_square_sum = matrix_square + matrix_square.T
        matrix_dot = np.dot(self.points.expect_outside_points_array,self.points.expect_outside_points_array.T)
        matrix_minus_squere_sum = matrix_square_sum - 2 * matrix_dot
        kdd = -D_constant * (matrix_minus_squere_sum + 2 * c ** 2)/(matrix_minus_squere_sum + c ** 2) ** 1.5 + Sigma_r * (matrix_minus_squere_sum + c ** 2) ** 0.5

        self.KDD = kdd
        return(self.KDD)

    def CalculateKDE(self):

        c = self.c

        matrix_square_expect_outside_points = np.sum(self.points.expect_outside_points_array ** 2, axis=1).reshape(EXPECT_OUTSIDE_POINTS_NUMBER,1)
        matrix_square_outside_points = np.sum(self.points.outside_points_array ** 2, axis=1).reshape(1,OUTSIDE_POINTS_NUMBER)
        matrix_square_sum = matrix_square_expect_outside_points + matrix_square_outside_points
        matrix_dot = np.dot(self.points.expect_outside_points_array,self.points.outside_points_array.T)
        matrix_minus_squere_sum = matrix_square_sum - 2 * matrix_dot
        kde = -D_constant * (matrix_minus_squere_sum + 2 * c ** 2)/(matrix_minus_squere_sum + c ** 2) ** 1.5 + Sigma_r * (matrix_minus_squere_sum + c ** 2) ** 0.5
        self.KDE = kde
        return(self.KDE)

    def CalculateKBD(self): 
        KBD = np.zeros((SURFACE_POINTS_NUMBER, EXPECT_OUTSIDE_POINTS_NUMBER))
        c = self.c
        for i in range(SURFACE_POINTS_NUMBER):
            for j in range(EXPECT_OUTSIDE_POINTS_NUMBER):
                if i < len(self.points.left_points):
                    xi = self.points.left_points[i].x
                    yi = self.points.left_points[i].y
                    xj = self.points.expect_outside_points[j].x
                    yj = self.points.expect_outside_points[j].y

                    KBD[i][j] = ((xi - xj)/(((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5))
                    

                if len(self.points.left_points) <= i < (len(self.points.left_points) + len(self.points.down_points)):
                    xi = self.points.down_points[i - len(self.points.left_points)].x
                    yi = self.points.down_points[i - len(self.points.left_points)].y
                    xj = self.points.expect_outside_points[j].x
                    yj = self.points.expect_outside_points[j].y

                    KBD[i][j] = ((yi - yj)/(((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5))
                    

                if 0 <= (i - (len(self.points.left_points) + len(self.points.down_points))) < len(self.points.right_points):
                    xi = self.points.right_points[(i - (len(self.points.left_points) + len(self.points.down_points)))].x
                    yi = self.points.right_points[(i - (len(self.points.left_points) + len(self.points.down_points)))].y
                    xj = self.points.expect_outside_points[j].x
                    yj = self.points.expect_outside_points[j].y

                    KBD[i][j] = (((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5)

                if 0 <= (i - (len(self.points.left_points) + len(self.points.down_points) + len(self.points.right_points))) < len(self.points.up_points):
                    xi = self.points.up_points[(i - (len(self.points.left_points) + len(self.points.down_points) + len(self.points.right_points)))].x
                    yi = self.points.up_points[(i - (len(self.points.left_points) + len(self.points.down_points) + len(self.points.right_points)))].y
                    xj = self.points.expect_outside_points[j].x
                    yj = self.points.expect_outside_points[j].y

                    KBD[i][j] = (((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5)
        self.KBD = KBD
        return(KBD)

    def CalculateKBE(self):
        KBE = np.zeros((SURFACE_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER))
        c = self.c
        for i in range(SURFACE_POINTS_NUMBER):
            for j in range(OUTSIDE_POINTS_NUMBER):
                if i < len(self.points.left_points):
                    xi = self.points.left_points[i].x
                    yi = self.points.left_points[i].y
                    xj = self.points.outside_points[j].x
                    yj = self.points.outside_points[j].y

                    KBE[i][j] = ((xi - xj)/(((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5))
                    

                if len(self.points.left_points) <= i < (len(self.points.left_points) + len(self.points.down_points)):
                    xi = self.points.down_points[i - len(self.points.left_points)].x
                    yi = self.points.down_points[i - len(self.points.left_points)].y
                    xj = self.points.outside_points[j].x
                    yj = self.points.outside_points[j].y

                    KBE[i][j] = ((yi - yj)/(((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5))
                    

                if 0 <= (i - (len(self.points.left_points) + len(self.points.down_points))) < len(self.points.right_points):
                    xi = self.points.right_points[(i - (len(self.points.left_points) + len(self.points.down_points)))].x
                    yi = self.points.right_points[(i - (len(self.points.left_points) + len(self.points.down_points)))].y
                    xj = self.points.outside_points[j].x
                    yj = self.points.outside_points[j].y

                    KBE[i][j] = (((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5)

                if 0 <= (i - (len(self.points.left_points) + len(self.points.down_points) + len(self.points.right_points))) < len(self.points.up_points):
                    xi = self.points.up_points[(i - (len(self.points.left_points) + len(self.points.down_points) + len(self.points.right_points)))].x
                    yi = self.points.up_points[(i - (len(self.points.left_points) + len(self.points.down_points) + len(self.points.right_points)))].y
                    xj = self.points.outside_points[j].x
                    yj = self.points.outside_points[j].y

                    KBE[i][j] = (((xi - xj) ** 2 + (yi - yj) ** 2 + c ** 2) ** 0.5)
        self.KBE = KBE
        return(KBE)

    def CalculatePsimatrix(self): 
        Psi_matrix = np.zeros((EXPECT_OUTSIDE_POINTS_NUMBER,POINTS_NUMBER))
        c = self.c

        matrix_square = np.sum(self.points.points_array ** 2, axis=1).reshape(1,POINTS_NUMBER)
        matrix_square_expect_outside_points = np.sum(self.points.expect_outside_points_array ** 2, axis=1).reshape(EXPECT_OUTSIDE_POINTS_NUMBER,1)
        matrix_square_sum = matrix_square + matrix_square_expect_outside_points
        matrix_dot = np.dot(self.points.expect_outside_points_array,self.points.points_array.T)
        matrix_minus_squere_sum = matrix_square_sum - 2 * matrix_dot
        
        Psi_matrix = (matrix_minus_squere_sum +  c ** 2)**0.5
        self.Psi_matrix = Psi_matrix

        return(Psi_matrix)
    
    def CalculateAll(self):
        #time1 = time.time()
        self.CalculateKDD()
        #time2 = time.time()
        #print('KDD %f'%(time2-time1))

        self.CalculateKDE()
        #time3 = time.time()
        #print('KDE %f'%(time3-time2))


        self.CalculateKBD()
        #time4 = time.time()
        #print('KBD %f'%(time4-time3))

        self.CalculateKBE()
        #time5 = time.time()
        #print('KBE %f'%(time5-time4))

        self.coefficient_matrix = np.vstack((np.hstack((self.KDD, self.KDE)),np.hstack((self.KBD, self.KBE))))
        self.inv_coefficient_matrix = np.linalg.inv(self.coefficient_matrix)


        #time6 = time.time()
        #print('inv_coefficient_matrix %f'%(time6-time5))

        self.CalculatePsimatrix()    
        #time7 = time.time()
        #print('CalculatePsimatrix %f'%(time7-time6))

        return(self)    


def Solve(c,a,xi,yi,i,integrate_matrix_list):

    f = lambda y, x: (((x - xi) ** 2 + (y - yi) ** 2 + c ** 2) ** 0.5)
    result = integrate.dblquad(f,0,a,0,a)
    integrate_matrix_list[i] = [result[0]]


def IntegrateCoefficientMatrix(initialPoints):
    c = c_constant
    a = a_constant

    integrate_matrix_list = multiprocessing.Manager().list(range(POINTS_NUMBER))
    p = pool.Pool()
    for i in range(POINTS_NUMBER):
        xi = initialPoints.points[i].x
        yi = initialPoints.points[i].y
        
        p.apply_async(Solve, args=(c,a,xi,yi,i,integrate_matrix_list))
    p.close()
    p.join()
    integrate_matrix = np.array(integrate_matrix_list).T
    return(integrate_matrix)
       
class DrawAndData():
    points = Points()
    x_values = np.zeros((1,1))
    y_values = np.zeros((1,1))
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
            self.Z = self.Z + z_val[i] * ((X - self.x_values[i]) ** 2 + (Y - self.y_values[i]) ** 2 + 1) ** 0.5 

    def PlotRandomPoint(self):
    
        plt.scatter(self.x_values,self.y_values)
        plt.xlim((-10,60))
        plt.ylim((-10,60))
        file_name = FILE_NAME + '/Random.png'
        plt.savefig(file_name)

        plt.close()

    def PlotNeutronFlux(self,a_matrix):
        fig = plt.figure()
        ax = Axes3D(fig)

        X = np.arange(0, 50, 1)
        Y = np.arange(0, 50, 1)
        X, Y = np.meshgrid(X, Y)
        z_val = np.array(a_matrix)[:,0]

        ax.plot_surface(X, Y, self.Z)
        plt.show()
    
    def PlotHeat(self):
        plt.imshow(self.Z,interpolation = 'nearest' , cmap = 'jet')
        plt.colorbar()
        file_name = FILE_NAME + '/Heat.png'
        plt.savefig(file_name)

        plt.close()

    def FileOutput(self, lamda, iterationFlag):
        file_name = self.file_name + 'data.csv'

        with open(file_name,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = ["x:",self.x_var,self.x_mean,"y:",self.y_var,self.y_mean,"lamda:",lamda,'iteration flag',iterationFlag]
        
            csv_write.writerow(data_row)

    def LamdaOutput(self, lamda):
        
        file_name = FILE_NAME + '/lamda.csv'
        '''
        with open(file_name,'a+',newline='') as f:
            csv_write = csv.writer(f)
            data_row = [lamda]
        
            csv_write.writerow(data_row)
        '''
        np.savetxt(file_name, lamda,delimiter=',')

    

def main():   
    #initial_time = time.time()
    initial_points = PointProcess().setPoint(INNER_POINTS_NUMBER,SURFACE_POINTS_NUMBER,OUTSIDE_POINTS_NUMBER)

    draw = DrawAndData(initial_points)
    draw.PlotRandomPoint()

    #print('PointProcess %f'%(time.time() - initial_time))
    #time1 = time.time()
    matrix = matrixProcess(initial_points, c_constant).CalculateAll()
    #print('MatrixProcess %f'%(time.time() - time1))
    f_matrix_before = np.vstack((np.ones((EXPECT_OUTSIDE_POINTS_NUMBER,1)),np.zeros((OUTSIDE_POINTS_NUMBER,1))))
    
    x = x_constant
    lamda = init_lamda
    lamda_before = init_lamda
    #time2 = time.time()
    integrate_coefficient_matrix = IntegrateCoefficientMatrix(initial_points)

    lamda_array = np.array([lamda])

    #print('solve Process %f'%(time.time() - time2))
    a_matrix_before = np.ones((POINTS_NUMBER,1))

    iteration_flag = False

    for i in range(10000):
        a_matrix = x / lamda * np.dot(matrix.inv_coefficient_matrix, f_matrix_before)  #A matrix

        f_matrix_no_zeros = nu * Sigma_f * np.dot(matrix.Psi_matrix, a_matrix)
        f_matrix = np.vstack((f_matrix_no_zeros,np.zeros((OUTSIDE_POINTS_NUMBER,1))))

        lamda = lamda_before *(np.dot(integrate_coefficient_matrix, a_matrix) / np.dot(integrate_coefficient_matrix, a_matrix_before))
        

        if np.isnan(np.min(a_matrix)) or lamda == 0:
            #np.nan_to_num(a_matrix)
            lamda_array = np.append(lamda_array, [[np.NaN]])
            break
        
        lamda_array = np.append(lamda_array, lamda[0][0])
        #print(lamda)

        #draw.LamdaOutput(lamda[0][0])
        '''
        lamda_output = threading.Thread(target=draw.LamdaOutput,args=(lamda[0][0],))
        lamda_output.start()
        lamda_output.join()        
        '''

        f_matrix_before = f_matrix
        a_matrix_before = a_matrix
        if abs(lamda_before - lamda) < 0.00001:
            iteration_flag =True
            break
        lamda_before = lamda
    
    draw.LamdaOutput(lamda_array)

    draw.CalculateZ(a_matrix)
    draw.PlotHeat()
    draw.FileOutput(lamda[0][0],iteration_flag)

def FlushConstant(innerPointsNumber,surfacePointsNumber,times):
    global INNER_POINTS_NUMBER
    global SURFACE_POINTS_NUMBER 
    global OUTSIDE_POINTS_NUMBER
    global EXPECT_OUTSIDE_POINTS_NUMBER
    global POINTS_NUMBER
    global FILE_NAME

    INNER_POINTS_NUMBER = INNER_POINTS_NUMBER_LIST[i]
    SURFACE_POINTS_NUMBER = SURFACE_POINTS_NUMBER_LIST[j]
    OUTSIDE_POINTS_NUMBER = SURFACE_POINTS_NUMBER_LIST[j]
    EXPECT_OUTSIDE_POINTS_NUMBER = INNER_POINTS_NUMBER + SURFACE_POINTS_NUMBER
    POINTS_NUMBER = INNER_POINTS_NUMBER + SURFACE_POINTS_NUMBER + OUTSIDE_POINTS_NUMBER
    if not os.path.exists('data/a_%d_inner_%d_outside_%d_c_%d/%d'%(a_constant, INNER_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER, c_constant,times)):
            os.makedirs('data/a_%d_inner_%d_outside_%d_c_%d/%d'%(a_constant, INNER_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER, c_constant,times))

    FILE_NAME = 'data/a_%d_inner_%d_outside_%d_c_%d/%d'%(a_constant, INNER_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER, c_constant,times)


INNER_POINTS_NUMBER_LIST = [c for c in range(50,1000,50)] + [c for c in range(1000,4100,100)]
SURFACE_POINTS_NUMBER_LIST = [10] + [c for c in range(25,100,25)] + [c for c in range(100,4050,50)]
for i in range(len(INNER_POINTS_NUMBER_LIST)):
    for j in range(len(SURFACE_POINTS_NUMBER_LIST)):
        if 7 * INNER_POINTS_NUMBER_LIST[i] < SURFACE_POINTS_NUMBER_LIST[j]:
            break
        for times in range(50):     
                if __name__ == '__main__':

                    FlushConstant(INNER_POINTS_NUMBER_LIST[i],SURFACE_POINTS_NUMBER_LIST[j],times)           
                    initial_time = time.time()
                    main()
                    finish_time = time.time()
                    print('a = %d, inner = %d, outside = %d, c = %d, times = %d , time = %fs'%
                         (a_constant, INNER_POINTS_NUMBER, OUTSIDE_POINTS_NUMBER, c_constant, (times + 1), (finish_time - initial_time)))
