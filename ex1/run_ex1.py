#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exercise:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

from ex1.compute_cost import compute_cost
from ex1.gradient_descent import gradient_descent
from plot_data import *


def run_ex1():
    plotting_tasks = Queue()  # create queue to send plotting data to new process
    tasks_that_are_done = Queue()
    processes = []

    # ==================== Part 1: Basic Function ====================

    print("Starting exercise 1.\n")

    # ======================= Part 2: Plotting =======================

    col_names = ['x0', 'y']
    data = pd.read_csv('datasets/ex1data1.txt', sep=",", names=col_names, header=None)
    # print(data)

    x = data.iloc[:, :1]  # vector X with samples
    y = data.iloc[:, 1]  # vector y with labels
    m = y.size  # size of training set
    print("{} samples found in dataset.".format(m))
    plotting_tasks.put("SCATTER")
    plotting_tasks.put(data)  # add data to the plotting queue
    # creating processes
    p1 = Process(target=plot_data, args=(plotting_tasks, tasks_that_are_done, 1))
    processes.append(p1)  # add new process to the processes list
    p1.start()  # start the new process
    time.sleep(.5)

    #  =================== Part 3: Cost and Gradient descent ===================
    input("Press Enter to continue...")

    x.insert(loc=0, column="bias", value=np.ones(m), allow_duplicates=True)  # add bias column to x
    #  print(x)

    theta = np.zeros(2)  # initialize fitting parameters

    print('\nTesting the cost function...\n')
    J = compute_cost(x, y, theta, m)  # compute and display initial cost
    print('With theta = [0 ; 0], Cost computed = {}\n'
          'Expected cost value (approx) 32.07\n'.format(J))

    # further testing of the cost function
    J = compute_cost(x, y, np.array([-1, 2]), m)
    print('With theta = [-1 ; 2], Cost computed = {}\n'
          'Expected cost value (approx) 54.24\n'.format(J))

    input("Press Enter to continue...")

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    print('Running Gradient Descent...\n')
    theta = gradient_descent(x, y, theta, m, alpha, iterations)  # run gradient descent

    # print theta to screen
    print('Theta found by gradient descent:{} ; {}\n'
          'Expected theta values (approx): -3.6303 ; 1.1664'.format(theta[0], theta[1]))

    # plot the linear regression fit
    plotting_tasks.put("SCATTER_AND_LINE")
    plotting_tasks.put(x)  # add x to the plotting queue, we don't need the bias column
    plotting_tasks.put(y)  # add y to the plotting queue
    plotting_tasks.put(theta)  # add theta to the plotting queue
    # creating new process
    p2 = Process(target=plot_data, args=(plotting_tasks, tasks_that_are_done, 2))
    processes.append(p2)  # add new process to the processes list
    p2.start()  # start the new process
    time.sleep(.5)

    # input("Press Enter to plot linear regression fit.")
    input("Press Enter to continue...")

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = (np.array([1, 3.5])).dot(theta)
    print("For population = 35,000, we predict a profit of {}".format(predict1 * 10000))
    predict2 = (np.array([1, 7])).dot(theta)
    print("For population = 70,000, we predict a profit of {}".format(predict2 * 10000))

    input("Press Enter to continue...")

    #  ============= Part 4: Visualizing J(theta_0, theta_1) =============

    print('Press Enter to plot the Cost Function J as a surface.')

    # Grid over which we will calculate J
    theta0_values = np.linspace(-10, 10, num=100)
    theta1_values = np.linspace(-1, 4, num=100)

    # initialize J_values to a matrix of 0's
    J_values = np.zeros((theta0_values.size, theta1_values.size))

    # fill out J_values
    for i in range(len(theta0_values)):
        for j in range(len(theta1_values)):
            t = [theta0_values[i], theta1_values[j]]
            J_values[i, j] = compute_cost(x, y, t, m)

    # plot the Cost Function J surface
    plotting_tasks.put("SURFACE-3D")
    plotting_tasks.put(theta0_values)  # add Theta 0 to the plotting queue
    plotting_tasks.put(theta1_values)  # add Theta 1  to the plotting queue
    plotting_tasks.put(J_values)  # add J values to the plotting queue
    # creating new process
    p3 = Process(target=plot_data, args=(plotting_tasks, tasks_that_are_done, 3))
    processes.append(p3)  # add new process to the processes list
    p3.start()  # start the new process
    time.sleep(.5)

    input("Press Enter to plot the Cost Function J as contour.")

    # plot the Cost Function J contour
    plotting_tasks.put("CONTOUR")
    plotting_tasks.put(theta0_values)  # add Theta0_values to the plotting queue
    plotting_tasks.put(theta1_values)  # add Theta1_values  to the plotting queue
    plotting_tasks.put(J_values)  # add J values to the plotting queue
    plotting_tasks.put(theta[0])  # add Theta0  to the plotting queue
    plotting_tasks.put(theta[1])  # add Theta1  to the plotting queue
    # creating new process
    p4 = Process(target=plot_data, args=(plotting_tasks, tasks_that_are_done, 4))
    processes.append(p4)  # add new process to the processes list
    p4.start()  # start the new process
    time.sleep(.5)

    input("Press Enter to continue.")

    # completing process, in practice it'll not do much because
    # current processes are blocked in plt.show()
    #  for p in range(len(processes)):
    #    plotting_tasks.put("END")
    #    time.sleep(2)

    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print("{}".format(tasks_that_are_done.get()))

    print("Exercise 1 completed.")

    return 0


if __name__ == '__main__':
    run_ex1()
