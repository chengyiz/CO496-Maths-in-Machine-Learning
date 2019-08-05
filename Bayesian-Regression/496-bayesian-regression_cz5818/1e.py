import numpy as np
import answers
import scipy.optimize
import matplotlib.pyplot as plt

N = 25
interval = [-0.5, 1]
X = np.reshape(np.linspace(interval[0],interval[1], N), (N, 1))
Y = np.cos(10*X**2) + 0.1 * np.sin(100*X)

alpha = 1.0
beta = 0.1

def generate_matrix(X,order):
    means = np.linspace(-0.5,1,order)
    scale = 0.1
    
    gaussian = lambda x, mean, scale: np.exp(-(x-mean)**2/(2*scale**2))
    result = []
    for data in X:
        x = data[0]
        row = [1]
        row.extend([gaussian(x, mean, scale) for mean in means])
        result.append(row)
    return np.array(result)

def gd(init, tol, iteration, order):
    Phi = generate_matrix(X, order)
    result = [init]
    guess_eval = -answers.lml(init[0], init[1], Phi, Y)
    guess_grad = -answers.grad_lml(init[0], init[1], Phi, Y)
    guess_grads = [guess_grad]
    guess_evals = [guess_eval]
    
    step = 0.001
    while (step*guess_grad[0] >= init[0] or step*guess_grad[1]+1e-2 >= init[1]):
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

        if i % 500 ==0:
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

        step = 0.01
        while (step*guess_grad[0] >= guess[0] or step*guess_grad[1] >= guess[1]):
            step *= 0.5

        tmp = -answers.lml(guess[0] - step*guess_grad[0], \
                           guess[1] - step*guess_grad[1],  Phi, Y)
        while (tmp > guess_eval):
            step *= 0.5
            tmp = -answers.lml(guess[0] - step*guess_grad[0], \
                               guess[1] - step*guess_grad[1],  Phi, Y)
        step_size = step

    print("iter={}, func val={}, alpha={}, beta={}".format(i,guess_eval, guess[0], guess[1]))
    return [np.array(result), guess_eval]

tol = 10**-11
order = 12
init = np.array([alpha, beta])
result = gd(init, tol, 50000, order)
alpha, beta = result[0][-1][0], result[0][-1][1]

Phi = generate_matrix(X, order)

Sn = np.linalg.inv(np.eye(Phi.shape[1])/alpha + np.dot(Phi.T, Phi)/beta)
Mn = np.dot(Sn, np.dot(Phi.T, Y) / beta)[:,0]

samples = np.random.multivariate_normal(Mn, Sn, 5)
#print(samples)
num_tests = 300
tests = np.linspace(-1, 1.5, num_tests)
expect = np.cos(10*tests**2) + 0.1*np.sin(100*tests)
test_matrix = generate_matrix(np.reshape(tests, (num_tests, 1)), order)
means = np.array([np.dot(Mn, test) for test in test_matrix])
for i in range(len(samples)):
    omega = samples[i]
    fits = np.array([np.dot(omega, test) for test in test_matrix])
    plt.plot(tests, fits, label='sample '+ str(i+1))
    
plt.plot(tests, means, label='pred mean')
plt.plot(X, Y, 'ro', markersize=2)
#plt.plot(tests, expect, label='expectation')

std_var = np.array([np.dot(test, np.dot(Sn, test)) for test in test_matrix])
std_err = np.array([np.dot(test, Sn.dot(test)) + beta for test in test_matrix])

pred_upper = means + 2*std_var**0.5
pred_lower = means - 2*std_var**0.5
noise_upper = means + 2*std_err**0.5
noise_lower = means - 2*std_err**0.5

plt.fill_between(tests, pred_upper, pred_lower, where=pred_upper>=pred_lower, alpha=0.5)
plt.plot(tests, noise_upper, '-.', )
plt.plot(tests,noise_lower,'-.')

plt.xlabel('x',fontsize=16)
plt.ylabel('y',fontsize=16)
plt.title(r'{} training data in interval [{}, {}]{}with order {}, $\alpha$={:.6f}, $\beta$={:.6f}'.format(N,interval[0], interval[1], '\n',order, alpha, beta), fontsize=16)
plt.grid(alpha=0.2)
plt.legend(loc='best')
plt.show()