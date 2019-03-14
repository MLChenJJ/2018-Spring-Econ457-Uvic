import numpy as np
from numpy import append, array, diagonal, tril, triu
from numpy.linalg import inv
from scipy.linalg import lu
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
import warnings
from sympy import *
import sympy as sym
from matplotlib import pyplot as plt
import copy
import math
from scipy import optimize
import scipy as sp
import scipy.optimize as opt

def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    '''
    @param f (function): function to optimize, must return a scalar score
        and operate over a numpy array of the same dimensions as x_start
    @param x_start (numpy array): initial position
    @param step (float): look-around radius in initial step
    @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
        an improvement lower than no_improv_thr
    @max_iter (int): always break after this number of iterations.
        Set it to 0 to loop indefinitely.
    @alpha, gamma, rho, sigma (floats): parameters of the algorithm
        (see Wikipedia page for reference)
    return: tuple (best parameter array, best score)
    '''
    # init
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        print('...best so far:', best)

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res)-1)

        # reflection
        xr = x0 + alpha*(x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma*(x0 - res[-1][0])
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho*(x0 - res[-1][0])
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma*(tup[0] - x1)
            score = f(redx)
            nres.append([redx, score])
        res = nres

def mygolden(f,a, b, maxit = 1000, tol = 1/10000):
    alpha1 = (3 - np.sqrt(5)) / 2
    alpha2 = (np.sqrt(5) - 1) / 2
    if a > b:
        a, b = b, a
        
    x1 = a + alpha1 * (b - a)
    x2 = a + alpha2 * (b - a)

    f1, f2 = f(x1), f(x2)

    d = (alpha1 * alpha2)*(b - a) # initial d
    while d > tol:
        d = d * alpha2 # alpha2 is the golden ratio
        if f2 < f1: # x2 is new upper bound
            x2, x1 = x1, x1 - d
            f2, f1 = f1, f(x1)
        else:  # x1 is new lower bound
            x1, x2 = x2, x2 + d
            f1, f2 = f2, f(x2)
            
    if f1>f2:
        x = x2
    else:
        x = x1      
    return x    


def fun(alpha,p1,p2,I,bound):
    bta = 1-alpha
    
    f = lambda x: np.power(x,alpha)*np.power((I-p1*x)*1.0/p2,bta)
    x1 = mygolden(f, bound[0], bound[1])
    x2 = (I-p1*x1)*1.0/p2
    
    return [x1,x2]

#this function is for question a
def plot_pic(v):
    f = lambda x: np.power(x,0.5)*np.power((100-x),0.5)
    x = np.linspace(0,100,100)
    y = f(x)
    plt.figure()
    plt.plot(x,y)
    plt.scatter(v[0],v[1],c='r')
    plt.title('utility function')
    plt.show()



#from here, code for question (c) starts


def cal_utility():
    f = lambda x: np.power(x[0],0.5)*np.power(x[1],0.3)*np.power((100-3*x[0]-2*x[1]),0.2)
    
    result = optimize.fmin(lambda x: -f(x),[5,2])
    x1 = result[0]
    x2 = result[1]
    x3 = 100-3*x1-2*x2
    
    return [x1,x2,x3]



def g(X):
    x,y = X
    return np.power(x,0.5)*np.power(y,0.3)*np.power((100-3*x-2*y),0.2)
    #return ((x-1)**4 + 5 * (y-1)**2 - 2*x*y)

def plot_u(value):
    X_opt = value
    fig, ax = plt.subplots(figsize=(6, 4)) 
    x_ = np.linspace(0, 33, 100) 
    y_ = np.linspace(0, 50, 100) 
    #x_ = y_ = np.linspace(1, 4, 100)
    X, Y = np.meshgrid(x_, y_)
    #f = lambda x: -g(x)
    c = ax.contour(X, Y, g((X, Y)), 50) 
    ax.plot(X_opt[0], X_opt[1], 'r*', markersize=15) 
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)
    plt.colorbar(c, ax=ax) 
    plt.title('consumer\'s utility objective function')
    fig.tight_layout()
    plt.show()
    
def plot_x3():
    fig, ax = plt.subplots(figsize=(6, 4)) 
    x_ = np.linspace(0, 33, 100) 
    y_ = np.linspace(0, 50, 100) 
    #x_ = y_ = np.linspace(1, 4, 100)
    X, Y = np.meshgrid(x_, y_)
    f = lambda x: 100-3*x[0]-2*x[1]
    
    c = ax.contour(X, Y, f((X, Y)), 50) 
    #ax.plot(X_opt[0], X_opt[1], 'r*', markersize=15) 
    ax.set_xlabel(r"$x_1$", fontsize=18)
    ax.set_ylabel(r"$x_2$", fontsize=18)
    plt.colorbar(c, ax=ax) 
    plt.title('budget constraint objective function')
    fig.tight_layout()
    plt.show()
    
    
    return 0
    



    
if __name__ == '__main__':
    
    print('This is the answer of question (a):')
    result = fun(0.5,1,1,100,[0,100])
    print('x1* = {0}, x2* = {1}'.format(result[0],result[1]))
    #plot_pic(result)
    print('\n\n')
    
    print('This is the answer of question (b):')
    result = fun(0.25,1,1,100,[0,100])
    print('\t In the case alpha = {0}, p1={1}, p2 = {2}, I = {3}: x1* = {4}, x2* = {5}'.format(0.25,1,1,100,result[0],result[1]))
    #print('\n')
    
    result = fun(0.5,2,1,100,[0,50])
    print('\t In the case alpha = {0}, p1={1}, p2 = {2}, I = {3}: x1* = {4}, x2* = {5}'.format(0.5,2,1,100,result[0],result[1]))
    #print('\n')
    
    result = fun(0.5,1,1,200,[0,200])
    print('\t In the case alpha = {0}, p1={1}, p2 = {2}, I = {3}: x1* = {4}, x2* = {5}'.format(0.5,1,1,200,result[0],result[1]))
    print('\n\n')
    
    
    print('This is the answer of question (c):')
    #resolve consumer's utility maximum
    result = cal_utility()
    print('x1* = {0}, x2* = {1}, x3* = {2}'.format(result[0],result[1],result[2]))
    #plot the consumer's utility objective function
    #plot_u(result)
    plot_x3()
    
    
    
    
    
    
