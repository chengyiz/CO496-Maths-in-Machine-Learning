import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

N = 25
X = np.reshape(np.linspace(0,0.9,N),(N,1))
Y = np.cos(10*X**2) + 0.1*np.sin(100*X)

def ridge_fit():
    
    def generate_matrix(datas):
        means = np.linspace(0,1,20)
        scale = 0.1
        
        gaussian = lambda x, mean, scale: np.exp(-(x-mean)**2/(2*scale**2))
        result = []
        for data in datas:
            x = data[0]
            row = [1]
            row.extend([gaussian(x, mean, scale) for mean in means])
            result.append(row)
        return np.array(result)
            
    def loss(train_matrix, train_result, theta, lam):
        epsilon = train_result[:,0] - np.dot(train_matrix, theta)
        return np.dot(epsilon, epsilon) + lam * np.dot(theta, theta)
        
        
    def map(train_matrix, train_result, lam):
        loss = lambda theta: lam * np.dot(theta, theta) + sum([(np.dot(train_matrix[i], theta) - train_result[i][0])**2 for i in range(train_matrix.shape[0])])
        ans = scipy.optimize.minimize(loss, np.ones(21))['x']
        opt_theta = np.reshape(np.linalg.solve(np.dot(train_matrix.T, train_matrix) + lam * np.eye(train_matrix.shape[1]), np.dot(train_matrix.T, train_result)), 21)
        if np.dot(ans - opt_theta, ans - opt_theta) > 1:
            print("++++++++WARNING+++++")
        return [ans, opt_theta]    
    
    def test_fit(test_data, theta):
        trans = generate_matrix(test_data)
        return trans.dot(theta)   
    
    train_matrix = generate_matrix(X)
    
    num_tests = 200
    tests = np.reshape(np.linspace(-0.3,1.3, num_tests), (num_tests,1))
    
    plt.ylim(top=1.5,bottom=-1.5)
    plt.plot(X,Y, 'go', markersize=4, label='data')
    colors = ['b','r','y']
    lam = [0,0.1,10]
    for i in range(3):
        test_result = test_fit(tests, map(train_matrix, Y, lam[i])[1])
        plt.plot(tests, test_result,color=colors[i],label='$\lambda$='+str(lam[i]))
    plt.legend(loc="lower left")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("gaussian basis function")
    plt.show()
    
if __name__ == "__main__":
    ridge_fit()