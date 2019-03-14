import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math
from cvxopt import matrix, solvers


c = np.array([0.10,0.20,0.15])
H = np.matrix([[0.005,-0.010,0.004],
               [-0.010,0.040,-0.002],
               [0.004,-0.002,0.023]])

def objective(x):
    return x@H@x


def constraint_mini_v(x):
    return 1-np.sum(x)
    

def minimum_variance(n):
    x0 = np.zeros(n)
    x0[0] = 1.0/n
    x0[1] = 1.0/n
    x0[2] = 1.0/n
    
    
    b = (0,np.inf)
    bnds = (b,b,b)
    con = {"type":"eq", 'fun':constraint_mini_v}
    solution = minimize(objective,x0,method ='SLSQP',bounds = bnds, constraints=con)
    x = solution.x
    
    e_return = x@c
    risk = math.sqrt(x@H@x)
    
    #return the rate of return of the portfolio with the minimum global variance
    return e_return


#constraint for highest_return
def constraintsum(x):
    return 1-np.sum(x)

#constraint for highest_return
def constraintmax(x):
    maxret = max(c)
    return x@c-maxret
    
def highest_return(n):
    x0 = np.zeros(n)
    x0[0] = 1.0/n
    x0[1] = 1.0/n
    x0[2] = 1.0/n
    
    #maxret = max(c)
    
    b = (0,np.inf)
    bnds = (b,b,b)
    conmax1 = {'type':'eq','fun':constraintsum}
    conmax2 = {'type':'eq','fun':constraintmax}
    conmax = ([conmax1,conmax2])
    
    solution = minimize(objective,x0,method='SLSQP',bounds = bnds,constraints = conmax)  
    x = solution.x  
    e_return = x@c 
    risk = math.sqrt(x@H@x)
    
    return e_return
    
    
def cvxopt_cal(c):
    Q = 2*matrix(H)
    p = matrix(c)
    G = matrix([[-1.0,0.0, 0.0],[0.0,-1.0, 0.0],[0.0, 0.0,-1.0 ]])
    # inequality constraint, right hand side
    h = matrix([0.0,0.0, 0.0])
    # equality constraint, left hand side shape = (1,3)
    A = matrix([1.0, 1.0, 1.0], (1,3))
    # equality constraint, right hand side
    b = matrix(1.0)
    
    sol=solvers.qp(Q, p, G, h, A, b)
    
    
    return sol
    
    
    
def pic_a(risk,points):
    y = risk
    x = points
    fig = plt.figure('Question a')
    plt.grid(True)
    ax1 = fig.add_subplot(111)
    
    plt.xlim(minimum_v,highest_r)
    ax1.set_title('Part a of question 1')
    plt.xlabel('expected rate of return')
    plt.ylabel('risk')
    ax1.scatter(x,y,c='b',marker='*')
    #plt.legend('x1')
    plt.show()
    
def pic_b(risk,points):
    x = risk
    y = points
    fig = plt.figure('Question b')
    #plt.grid(True)
    ax1 = fig.add_subplot(111)
    #plt.plot(x,y)
    plt.ylim(minimum_v,highest_r)
    ax1.set_title('Part b of Question 1')
    plt.ylabel('expected rate of return')
    plt.xlabel('risk')
    ax1.scatter(x,y,c='r',marker='*')
    #plt.legend('x1')
    plt.show()
    
    
    
    
    


if __name__ == '__main__':
    minimum_v = minimum_variance(3)
    highest_r = highest_return(3)
    bnds = [minimum_v,highest_r]
    
    begin = minimum_v
    
    points = []
    risk = []
    
    for i in range(20):
        value = begin+((highest_r-minimum_v)/20)*i
        points.append(value)
    points.append(highest_r)
    
    
    num = 0
    count = 0
    while(num<7):
        c = np.array([points[count],points[count+1],points[count+2]])
        result = cvxopt_cal(c)
        value = result['x']
        for item in value:
            risk.append(item)
        count = count+3
        num = num+1
    print(points)
    print(risk)
    
    
    pic_a(risk, points)
    pic_b(risk, points)
    
    

    
    
    
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 