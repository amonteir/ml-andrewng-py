"""
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear regression exercise.
%
%  You will need to complete the following functions in this
%  exercise:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
"""

from ex1.compute_cost import compute_cost
from ex1.gradient_descent import gradient_descent, gradient_descent_multi
from plot_data import *
from multiprocessing import Process, Queue
import pandas as pd
from feature_normalise import feature_normalise


def run_ex1_multi():
    #  Initialization
    plotting_tasks = Queue()  # create queue to send plotting data to new process
    tasks_that_are_done = Queue()
    processes = []

    #  ================ Part 1: Feature Normalization ================

    print("Starting exercise 1 Multi.\n")

    # ======================= Part 2: Plotting =======================

    col_names = ['x0', 'x1', 'y']
    data = pd.read_csv('datasets/ex1data2.txt', sep=",", names=col_names, header=None)
    # print(data)

    x = data.loc[:, ['x0', 'x1']]  # vector X with samples
    y = data.loc[:, 'y']  # vector y with labels
    m = y.size  # size of training set
    y = y.values.reshape(-1, 1)  # convert array to dataframe
    print("{} samples found in dataset.".format(m))

    # Scale features and set them to zero mean
    print('Normalizing Features...')

    x, mu, sigma = feature_normalise(x)

    print("mu = [{} {}], sigma = [{} {}]".format(mu[0, 0], mu[0, 1], sigma[0, 0], sigma[0, 1]))
    print(x.iloc[0:10, :])

    x.insert(loc=0, column="bias", value=np.ones(m), allow_duplicates=True)  # add bias column to x

    """
    %% ================ Part 2: Gradient Descent ================

    % ====================== YOUR CODE HERE ======================
    % Instructions: We have provided you with the following starter
    %               code that runs gradient descent with a particular
    %               learning rate (alpha). 
    %
    %               Your task is to first make sure that your functions - 
    %               computeCost and gradientDescent already work with 
    %               this starter code and support multiple variables.
    %
    %               After that, try running gradient descent with 
    %               different values of alpha and see which one gives
    %               you the best result.
    %
    %               Finally, you should complete the code at the end
    %               to predict the price of a 1650 sq-ft, 3 br house.
    %
    % Hint: At prediction, make sure you do the same feature normalization.
    %
    """

    print('Running gradient descent...')

    # Choose some alpha value
    alpha = 0.1
    num_iterations = 400

    # Init Theta and Run Gradient Descent
    theta = np.zeros((3, 1))
    theta, J_history = gradient_descent_multi(x, y, theta, m, alpha, num_iterations)
    # print(f"Theta {theta}, \nJ_history: {J_history}")

    # Plot the convergence graph
    plotting_tasks.put("LINE")
    plotting_tasks.put(np.arange(num_iterations))  # add iters axis
    plotting_tasks.put(J_history)  # add y to the plotting queue

    # creating new process
    num_processes = 1
    p1 = Process(target=plot_data, args=(plotting_tasks, tasks_that_are_done, num_processes))
    processes.append(p1)  # add new process to the processes list
    p1.start()  # start the new process
    time.sleep(.5)
    num_processes += 1

