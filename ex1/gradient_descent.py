"""
Performs gradient descent to learn theta
   theta = gradient_descent(X, y, theta, alpha, num_iters) updates theta by
   taking num_iters gradient steps with learning rate alpha

"""
import numpy as np
from ex1.compute_cost import compute_cost, compute_cost_multi


def gradient_descent(x, y, theta, m, alpha, num_iterations):
    J_history = np.zeros(num_iterations)  # init history vector for cost in each iteration

    for i in range(num_iterations):
        #  x.dot(theta) - y
        theta = theta - (alpha / m) * x.T.dot((x.dot(theta) - y))
        print('n={}, GD Theta0={}, GD Theta1={}'.format(i, theta[0], theta[1]))
        J_history[i] = compute_cost(x, y, theta, m)

    return theta


def gradient_descent_multi(x, y, theta, m, alpha, num_iterations):
    J_history = np.zeros((num_iterations, 1))  # init history vector for cost in each iteration

    for i in range(num_iterations):
        """
        a = x.dot(theta)  # 47x3 * 3x1 = 47x1
        b = a - y  # 47x1 - 47x1 = 47x1
        c = x.T.dot(b)  # 3x47 * 47x1 = 3x1
        d = (alpha/m) * c  # 3x1
        theta = theta - d  # 3x1 - 3x1 = 3x1
        """
        theta = theta - (alpha / m) * x.T.dot((x.dot(theta) - y))

        J_history[i, 0] = compute_cost_multi(x, y, theta, m)

    return theta, J_history
