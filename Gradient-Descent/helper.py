import answers
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import time

B = np.array([[3,-1],[-1,3]])
a = np.array([1,0])
b = np.array([0,-1])



def f2():
    start = time.time()
    density = 300
    x = np.linspace(-0.9,1.07,density)
    y = np.linspace(-1.7,-0.5, density)
    X, Y = np.meshgrid(x, y)
    
    step_size = 0.05
    
    func_2 = lambda x : np.sin(np.dot(x-a,x-a)) + np.dot(x-b, np.dot(B, x-b))
    Z = np.zeros([density, density])
    for i in range(density):
        for j in range(density):
            Z[i,j] = func_2(np.array([X[i,j], Y[i,j]]))
    result_2 = scipy.optimize.minimize(func_2, np.array([1,-1]))
    print(result_2)
    xs = np.zeros([51,2])
    xs[0] = np.array([1,-1])
    
    for i in range(50):
        direction = answers.grad_f2(xs[i])
        xs[i+1] = xs[i] - step_size*direction
    print(xs[-1])
    plt.contour(X,Y,Z,150)
    plt.plot(xs[:,0], xs[:,1],'ro',markersize=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("gradient_descent_f2 with step size {}".format(step_size))
    duration = time.time() - start
    print("total duration: {0:.6f}".format(duration))    
    plt.show()

def func_3(x):
    bla = -np.dot(x-a, x-a)
    tmp_1 = np.exp(bla)
    tmp_2 = np.exp(-np.dot(x-b, np.dot(B,x-b)))
    tmp_3 = 0.1 * np.log(np.linalg.det(np.eye(2) * 0.01 + np.outer(x,x)))
    return 1 - (tmp_1 + tmp_2 - tmp_3)

def f3():
    start = time.time()
    density = 300
    x = np.linspace(-0.1,1.2,density)
    y = np.linspace(-1.2,0.2, density)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros([density, density])
    for i in range(density):
        for j in range(density):
            Z[i,j] = func_3(np.array([X[i,j], Y[i,j]]))    
    result_3 = scipy.optimize.minimize(func_3, np.array([1,-1]))
    print(result_3)
    step_size = 0.2
    xs = np.zeros([51,2])
    xs[0] = np.array([1,-1])
    for i in range(50):
        direction = answers.grad_f3(xs[i])
        xs[i+1] = xs[i] - step_size*direction
    print(xs[-1])
    plt.contour(X,Y,Z,100)
    plt.plot(xs[:,0], xs[:,1],'ro',markersize=2)
    plt.xlabel("x")
    plt.ylabel("y")    
    plt.title("gradient_descent_f3 with step size 0.2")
    duration = time.time() - start
    print("total duration: {0:.6f}".format(duration))    
    plt.show()    

def g2_min():
    start = time.time()
    density = 1000
    x = np.linspace(-1.8,1,density)
    y = np.linspace(-2.8,0, density)
    X, Y = np.meshgrid(x, y)
    func_2 = lambda x : np.sin(np.dot(x-a,x-a)) + np.dot(x-b, np.dot(B, x-b))
    Z = np.zeros([density, density])
    for i in range(density):
        for j in range(density):
            Z[i,j] = func_2(np.array([X[i,j], Y[i,j]]))
    plt.contour(X,Y,Z,1000)
    plt.title("f2 minima")
    duration = time.time() - start
    print("total duration: {0:.6f}".format(duration))    
    plt.show()
    
    
def g3_min():
    start = time.time()
    density = 500
    x = np.linspace(-0.25,1.2,density)
    y = np.linspace(-1.3,0.4, density)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros([density, density])
    for i in range(density):
        for j in range(density):
            Z[i,j] = func_3(np.array([X[i,j], Y[i,j]]))
    plt.contour(X,Y,Z,1000)
    plt.title("f3 minima")
    duration = time.time() - start
    print("total duration: {0:.6f}".format(duration))    
    plt.show()    

def g2_step():
    start = time.time()
    func_2 = lambda x : np.sin(np.dot(x-a,x-a)) + np.dot(x-b, np.dot(B, x-b))
    col = 3
    row = 3
    steps= [0.1,0.2,0.3,0.45,0.6,0.8,1,1.1,1.2]
    fig = plt.figure(figsize = (8,8))
    for i in range(1, col*row + 1):
        print("{}-th image".format(i))
        stamp = time.time()
        step_size = steps[i-1]
        xs = np.zeros([51,2])
        xs[0] = np.array([1,-1])
        for j in range(50):
            direction = answers.grad_f2(xs[j])
            new = xs[j] - step_size*direction
            xs[j+1] = new
            
        left, right = max(min(xs[:,0]-1),-np.abs(xs[4,0])-1), min(np.abs(xs[4,0])+1, max(xs[:,0]+1))
        below, above = max(-np.abs(xs[4,1])-1,min(xs[:,1]-1)), min(np.abs(xs[4,1])+1, max(xs[:,1]+1))
        xs = xs[xs[:,0]<right]
        xs = xs[xs[:,0]>left]
        xs = xs[xs[:,1]>below]
        xs = xs[xs[:,1]<above]
        density = 300
        x = np.linspace(left,right,density)
        y = np.linspace(below,above, density)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros([density, density])
        for k in range(density):
            for j in range(density):
                Z[k,j] = func_2(np.array([X[k,j], Y[k,j]]))  
        fig.add_subplot(row, col, i)
        plt.contour(X,Y,Z,300)        
        plt.plot(xs[:,0], xs[:,1],'ro',markersize=2)
        plt.title("gd_f2 with step size {}".format(round(step_size,2)))
        print("partial duration: {0:.5f}".format(time.time()-stamp))
    duration = time.time() - start
    print("total duration: {0:.6f}".format(duration))
    plt.tight_layout()
    plt.show()

def g3_step():
    start = time.time()
    col = 3
    row = 3
    steps= [0.1,0.2,0.3,0.45,0.6,0.8,1,1.1,1.2]
    sol = np.array([ 0.11219288, -0.85537636])
    fig = plt.figure(figsize = (8,8))
    for i in range(1, col*row + 1):
        print("{}-th image".format(i))
        stamp = time.time()
        step_size = steps[i-1]
        xs = np.zeros([101,2])
        xs[0] = np.array([1,-1])
        for j in range(100):
            direction = answers.grad_f3(xs[j])
            new = xs[j] - step_size*direction
            xs[j+1] = new
            
        left, right = -0.25,1.2
        below, above = -1.3,0.5
        density = 200
        x = np.linspace(left,right,density)
        y = np.linspace(below,above, density)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros([density, density])
        for k in range(density):
            for l in range(density):
                Z[k,l] = func_3(np.array([X[k,l], Y[k,l]]))  
        fig.add_subplot(row, col, i)
        plt.contour(X,Y,Z,200)        
        plt.plot(xs[:,0], xs[:,1],'ro',markersize=2)
        plt.title("gd_f3 with step size {}".format(round(step_size,2)))
        print("partial duration: {0:.5f}".format(time.time()-stamp))
    duration = time.time() - start
    print("total duration: {0:.6f}".format(duration))
    fig.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    f2()
    #f3()
    #g2_min()
    #g3_min()
    #g2_step()
    #g3_step()