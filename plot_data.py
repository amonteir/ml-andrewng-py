import logging
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Lock, Process, Queue, current_process
import queue  # imported for using queue.Empty exception
import time
import numpy as np
from matplotlib import ticker

def plot_data(tasks_to_accomplish, tasks_that_are_done, figure_num):
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            first_task = tasks_to_accomplish.get_nowait()
            if first_task == "END":
                break
            elif first_task == "SCATTER":
                data = tasks_to_accomplish.get_nowait()
                if isinstance(data, pd.DataFrame):
                    x = data.iloc[:, :1]  # vector X with samples
                    y = data.iloc[:, 1]  # vector y with labels
                    plt.figure(figure_num)
                    plt.scatter(x, y)
                    plt.title("Market Size - Training Dataset")
                    plt.xlabel("Population of City in 10,000s")
                    plt.ylabel("Profit in $10,000s")
                    plt.show()
                    tasks_that_are_done.put("Process {} closed Figure {}".format(current_process().name, figure_num))
                    time.sleep(.5)

            elif first_task == "SCATTER_AND_LINE":
                x = tasks_to_accomplish.get_nowait()
                y = tasks_to_accomplish.get_nowait()
                theta = tasks_to_accomplish.get_nowait()
                plt.figure(figure_num)
                plt.scatter(x.iloc[:, 1], y)  # plot training dataset
                plt.plot(x.iloc[:, 1], x.dot(theta), color='r')  # plot linear regression fit
                plt.title("Market Size - Liner Regression Fit")
                plt.xlabel("Population of City in 10,000s")
                plt.ylabel("Profit in $10,000s")
                plt.show()
                tasks_that_are_done.put("Process {} closed Figure {}".format(current_process().name, figure_num))
                time.sleep(.5)

            elif first_task == "SURFACE-3D":
                x = tasks_to_accomplish.get_nowait()  # theta0_values
                y = tasks_to_accomplish.get_nowait()  # theta1_values
                z = tasks_to_accomplish.get_nowait()  # Cost Function J
                fig = plt.figure(figure_num)
                ax = plt.axes(projection='3d')
                ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

                plt.title("Cost Function, J")
                plt.xlabel("Theta0")
                plt.ylabel("Theta1")

                plt.show()
                tasks_that_are_done.put("Process {} closed Figure {}".format(current_process().name, figure_num))
                time.sleep(.5)

            elif first_task == "CONTOUR":
                x = tasks_to_accomplish.get_nowait()  # theta0_values
                y = tasks_to_accomplish.get_nowait()  # theta1_values
                z = tasks_to_accomplish.get_nowait()  # Cost Function J
                theta0 = tasks_to_accomplish.get_nowait()  # theta0
                theta1 = tasks_to_accomplish.get_nowait()  # theta1
                x_mesh, y_mesh = np.meshgrid(x, y)

                fig, ax = plt.subplots(1, 1)
                # fig = plt.figure(figure_num)

                cp = ax.contourf(x_mesh, y_mesh, z, locator=ticker.LogLocator())
                plt.plot(theta0, theta1, 'x', color="r")

                plt.title("Cost Function, J")
                plt.xlabel("Theta0")
                plt.ylabel("Theta1")

                plt.show()
                tasks_that_are_done.put("Process {} closed Figure {}".format(current_process().name, figure_num))
                time.sleep(.5)
            else:
                continue  # go back up to the loop

        except queue.Empty:
            continue

    return True
