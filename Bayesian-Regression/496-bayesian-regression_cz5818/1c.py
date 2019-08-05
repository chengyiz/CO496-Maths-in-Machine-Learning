import numpy as np
import answers
import scipy.optimize
import matplotlib.pyplot as plt


N = 26
X = np.reshape(np.linspace(0, 0.9, N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

def generate_for_tri(lst, deg):
    result = []
    for i in range(lst.shape[0]):
        row = []
        for j in range(deg+1):
            if j == 0:
                row.append(1)
            else:
                tmp = 2 * np.pi * j * lst[i,0]
                row.append(np.sin(tmp))
                row.append(np.cos(tmp))
        result.append(row)
    return np.array(result)


def gd(init, tol, iteration, order):
    Phi = generate_for_tri(X, order)
    result = [init]
    guess_eval = -answers.lml(init[0], init[1], Phi, Y)
    guess_grad = -answers.grad_lml(init[0], init[1], Phi, Y)
    guess_grads = [guess_grad]
    guess_evals = [guess_eval]
    
    step = 0.01
    while (step*guess_grad[0] >= init[0] or step*guess_grad[1] >= init[1]):
        step *= 0.5
        
    tmp = -answers.lml(init[0] - step*guess_grad[0], \
                       init[1] - step*guess_grad[1],  Phi, Y)
    while (tmp > guess_eval):
        step *= 0.5
        tmp = -answers.lml(init[0] - step*guess_grad[0], \
                           init[1] - step*guess_grad[1],  Phi, Y)
    step_size = step    
    
    convsgd = []
    lenXgd = []
    diffFsd = []
    
    print("iter={}, func val={}, alpha={}, beta={}".format(0,guess_eval, init[0], init[1]))
    for i in range(1,iteration+1):
        #step_size /= np.sqrt(i)
        guess = result[i-1] - step_size * guess_grads[i-1]
        result.append(guess)
        
        guess_eval = -answers.lml(guess[0], guess[1], Phi, Y)
        guess_grad = -answers.grad_lml(guess[0], guess[1], Phi, Y)
        
        guess_evals.append(guess_eval)
        guess_grads.append(guess_grad)
        
        if guess_eval >= 0 and i % 1000 ==0:
            print("iter={}, func val={}, alpha={}, beta={}".format(i,guess_eval, guess[0], guess[1]))
        
        convsgd.append(np.linalg.norm(guess_grad))
        lenXgd.append(np.linalg.norm(result[-1]-result[-2]))
        diffFsd.append(np.abs(guess_evals[-1]-guess_evals[-2]))
        
        if convsgd[-1] <= tol:
            print("First-Order Optimality Condition met")
            break
        elif lenXgd[-1] <= tol:
            print("Design not changing")
            break
        #elif diffFsd[-1] <= 0:
            #print("Objective not changing")
            #break
        elif i+1 >= iteration:
            print("Done iterating")
            break
        
        step = 0.1
        while (step*guess_grad[0] > guess[0] or step*guess_grad[1] > guess[1]):
            step *= 0.5
            
        m = np.linalg.norm(guess_grad)
        tmp = -answers.lml(guess[0] - step*guess_grad[0], \
                           guess[1] - step*guess_grad[1],  Phi, Y)
        while (tmp > guess_eval - step*0.01*m):
            step *= 0.5
            tmp = -answers.lml(guess[0] - step*guess_grad[0], \
                               guess[1] - step*guess_grad[1],  Phi, Y)
        step_size = step

    print("iter={}, func val={}, alpha={}, beta={}".format(i,guess_eval, guess[0], guess[1]))
    return [np.array(result), guess_eval]

tol = 10**-12

inits = np.array([[0.1, 0.5], [0.2, 0.2], [0.2, 0.1], \
                  [0.1, 0.1], [0.1, 0.1], [0.1, 0.1], \
                  [0.1, 0.1], [0.1, 0.1], [0.15,0.15], \
                  [0.05,0.05], [0.05,0.05], [0.05,0.05], \
                  [0.05,0.05], [0.05,0.05], [0.05,0.05]])
init = np.array([0.15,0.15])
results = []
for order in range(12):
    print("--------------------order {}--------------------".format(order))
    results.append(gd(inits[order], tol, 50000, order))
    print(results[-1][0][-1])
    print(results[-1][1])

for result in results:
    print(result[0][-1])
    
lmls = np.array([-result[1] for result in results])
xaxis = np.linspace(0,11,12)
plt.plot(xaxis, lmls)
plt.xlabel('order')
plt.ylabel('log max likelihood')
plt.title('log max likelihood versus order ({} data)'.format(N))
#density = 200
#x = np.linspace(min(result[:,0]), max(result[:,0]), density)
#y = np.linspace(min(result[:,1]), max(result[:,1]), density)
#xaxis, yaxis = np.meshgrid(x, y)
#zaxis = np.zeros([density, density])
#for k in range(density):
    #for j in range(density):
        #zaxis[k,j] = answers.lml(xaxis[k,j], yaxis[k,j], Phi, Y)
#plt.contour(xaxis,yaxis, zaxis, 150)
#plt.plot(result[:,0], result[:,1])
#plt.xlabel(r'$\alpha$')
#plt.ylabel(r'$\beta$')
plt.show()