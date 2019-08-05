import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0,0.9,N),(N,1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def tri_fit():
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
    
    num_tests = 200


    def mle(order, training_matrix, training_result):
        loss = lambda theta: sum([(np.dot(train_matrix[i], theta) - training_result[i][0])**2 for i in range(train_matrix.shape[0])])
        ans = scipy.optimize.minimize(loss, np.ones(2*order+1))['x']
        opt_theta = np.reshape(np.linalg.solve(np.dot(train_matrix.T, train_matrix), np.dot(train_matrix.T, training_result)), 2*order+1)
        if np.dot(ans - opt_theta, ans - opt_theta) > 1:
            print("++++++++WARNING+++++")
        return [ans, opt_theta]
    
    def test_fit(test_data, order, theta):
        trans = generate_for_tri(test_data, order)
        return trans.dot(theta)
    
    sum_sigmas = []
    errors = []
    for order in range(12):
        print("-------------order "+str(order)+"-----------")
        sum_sigma=0
        error = 0
        for i in range(N):
            test_idx = i
            training_data = np.array([X[j] for j in range(N) if j != test_idx])
            test_data = np.array([X[test_idx]])
            training_result = np.array([Y[j] for j in range(N) if j != test_idx])
            test_result = np.array([Y[test_idx]])
            
            train_matrix = generate_for_tri(training_data, order)
            opt_theta = mle(order, train_matrix, training_result)[1]
            square_error = (test_result[0,0] - np.dot(generate_for_tri(test_data, order), opt_theta))**2
            
            epsilon = training_result[:,0] - np.dot(train_matrix, opt_theta)
            sigma = np.dot(epsilon, epsilon) / (N-1)
            error += square_error
            sum_sigma += sigma
        print(opt_theta)
        sum_sigmas.append(sum_sigma/N)
        errors.append(error/N)
    
    #plt.ylim(top=1,bottom=-0.2)
    plt.plot([i for i in range(0, 12)], np.array(sum_sigmas), 'r', label='$\sigma^2_{ML}$')
    plt.plot([i for i in range(0, 12)], np.array(errors), 'g', label='avg squared error')
    plt.legend(loc='best')
    plt.xlabel("order")
    plt.title("trigonometric cross validation ({} data)".format(N))
    plt.show()  
    
if __name__ == "__main__":
    tri_fit()