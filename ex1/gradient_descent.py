"""
Performs gradient descent to learn theta
   theta = gradient_descent(X, y, theta, alpha, num_iters) updates theta by
   taking num_iters gradient steps with learning rate alpha

"""
import pandas as pd
import numpy as np
from ex1.compute_cost import compute_cost

def gradient_descent(x, y, theta, m, alpha, num_iterations):

    J_history = np.zeros(num_iterations)  # init history vector for cost in each iteration

    for i in range(num_iterations):
        #  x.dot(theta) - y
        theta = theta - (alpha / m) * x.T.dot((x.dot(theta) - y))
        print('n={}, GD Theta0={}, GD Theta1={}'.format(i, theta[0], theta[1]))
        J_history[i] = compute_cost(x, y, theta, m)


    return theta
