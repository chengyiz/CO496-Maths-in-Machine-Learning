import numpy as np
import answers
import scipy.optimize
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(-0, 0.9, N), (N, 1))
func = lambda X: np.cos(10*X**2) + 0.1 * np.sin(100*X)
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
    
order = 10
Phi = generate_matrix(X, order)
Sn = np.linalg.inv(np.eye(Phi.shape[1])/alpha + np.dot(Phi.T, Phi)/beta)
Mn = np.dot(Sn, np.dot(Phi.T, Y) / beta)[:,0]

samples = np.random.multivariate_normal(Mn, Sn, 5)
print(samples)
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
#plt.plot(tests, func(tests), '-o', markersize=3)

plt.xlabel('x',fontsize=16)
plt.ylabel('y',fontsize=16)
plt.title('1d',fontsize=16)
plt.grid(alpha=0.2)
plt.legend(loc='best')
plt.show()