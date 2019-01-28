import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0,0.9,N),(N,1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def polynomial_fit():
    
    def generate_for_poly(lst, deg):
        result = [[lst[j,0]**i for i in range(deg+1)] for j in range(lst.shape[0])]
        return np.array(result)
    
    num_tests = 200
    tests = np.reshape(np.linspace(-0.3,1.3, num_tests), (num_tests,1))

    def mle(order):
        train_matrix = generate_for_poly(X, order)
        loss = lambda theta: sum([(np.dot(train_matrix[i], theta) - Y[i][0])**2 for i in range(train_matrix.shape[0])])
        ans = scipy.optimize.minimize(loss, np.ones(order+1))['x']
        opt_theta = np.reshape(np.linalg.solve(np.dot(train_matrix.T, train_matrix), np.dot(train_matrix.T, Y)), order+1)
        print(opt_theta)
        print(ans)
        return [ans, opt_theta]
    
    def test_fit(test_data, order, theta):
        trans = generate_for_poly(test_data, order)
        return trans.dot(theta)
    
    plt.ylim(top=7,bottom=-2)
    plt.plot(X,Y, 'go', markersize=4, label='data')
    colors = ['b','r','c','m','y']
    orders=[0,1,2,3,11]    
    for i in range(len(orders)):
        test_result = test_fit(tests, orders[i], mle(orders[i])[1])
        plt.plot(tests, test_result,color=colors[i],label='order_'+str(orders[i]))
    g = np.cos(10*tests**2)+0.1*np.sin(100*tests)
    plt.plot(tests, g, color='k', label='g(x)',alpha=0.5)    
    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("polynomial basis function")    
    plt.show()
    
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
    
    def mle(order):
        train_matrix = generate_for_tri(X, order)
        print(train_matrix.shape)
        loss = lambda theta: sum([(np.dot(train_matrix[i], theta) - Y[i][0])**2 for i in range(train_matrix.shape[0])])
        ans = scipy.optimize.minimize(loss, np.ones(2*order+1))['x']
        opt_theta = np.reshape(np.linalg.solve(np.dot(train_matrix.T, train_matrix), np.dot(train_matrix.T, Y)), 2*order+1)
        print(opt_theta)
        print(ans)
        return [ans, opt_theta]
    
    def test_fit(test_data, order, theta):
        trans = generate_for_tri(test_data, order)
        return trans.dot(theta)
    
    num_tests = 200
    tests = np.reshape(np.linspace(-1,1.2, num_tests), (num_tests,1))
    plt.ylim(top=2,bottom=-2)
    plt.plot(X,Y, 'go', markersize=4, label='data')
    colors = ['b','r']
    orders=[1,11]
    for i in range(len(orders)):
        print("--------order "+str(orders[i]) +"---------")
        test_result = test_fit(tests, orders[i], mle(orders[i])[1])
        plt.plot(tests, test_result,color=colors[i],label='order_'+str(orders[i]))
    g = np.cos(10*tests**2)+0.1*np.sin(100*tests)
    plt.plot(tests, g, color='y', label='g(x)',alpha=0.5)
    plt.legend(loc='best')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("trigonometric basis function")
    plt.show()    
    
    
    
if __name__ == "__main__":
    #polynomial_fit()
    tri_fit()