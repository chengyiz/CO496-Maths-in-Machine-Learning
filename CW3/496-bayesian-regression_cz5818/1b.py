import numpy as np
import answers
import scipy.optimize
import matplotlib.pyplot as plt


N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

def generate_for_poly(lst, deg):
    result = [[lst[j,0]**i for i in range(deg+1)] for j in range(lst.shape[0])]
    return np.array(result)

Phi = generate_for_poly(X, 1)
step_size = 0.015
def gd(init, tol, iteration):
    result = [init]
    guess_eval = -answers.lml(init[0], init[1], Phi, Y)
    guess_grad = -answers.grad_lml(init[0], init[1], Phi, Y)
    guess_grads = [guess_grad]
    guess_evals = [guess_eval]
    
    convsgd = []
    lenXgd = []
    diffFsd = []
    
    print("iter={}, func val={}".format(0,guess_evals[0]))
    for i in range(1,iteration):
        guess = result[i-1] - step_size * guess_grads[i-1]
        result.append(guess)
        
        guess_eval = -answers.lml(guess[0], guess[1], Phi, Y)
        guess_grad = -answers.grad_lml(guess[0], guess[1], Phi, Y)
        
        guess_evals.append(guess_eval)
        guess_grads.append(guess_grad)
        
        print("iter={}, func val={}".format(i,guess_evals[i]))
        
        convsgd.append(np.linalg.norm(guess_grad))
        lenXgd.append(np.linalg.norm(result[-1]-result[-2]))
        diffFsd.append(np.abs(guess_evals[-1]-guess_evals[-2]))
        
        if convsgd[-1] <= tol:
            print("First-Order Optimality Condition met")
            break
        #elif lenXgd[-1] <= tol:
            #print("Design not changing")
            #break
        #elif diffFsd[-1] <= tol:
            #print("Objective not changing")
            #break
        elif i+1 >= iteration:
            print("Done iterating")
            break
        
    return np.array(result)

tol = 10**-12

result = gd(np.array([0.46,0.46]), tol, 50000)
print(result[-1])
print(result.shape)

density = 150
x = np.linspace(0.41, 0.465, density)
y = np.linspace(0.44, 0.461, density)
xaxis, yaxis = np.meshgrid(x, y)
zaxis = np.zeros([density, density])
for k in range(density):
    for j in range(density):
        zaxis[k,j] = answers.lml(xaxis[k,j], yaxis[k,j], Phi, Y)
con = plt.contour(xaxis,yaxis, zaxis, 30)
plt.clabel(con, inline=1, fontsize=8)
plt.plot(result[:,0], result[:,1])
plt.xlabel(r'$\alpha$',fontsize=14)
plt.ylabel(r'$\beta$',fontsize=14)
plt.title('1-order polynomial with step size '+str(step_size))
plt.show()
        