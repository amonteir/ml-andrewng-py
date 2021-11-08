import logging
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
            if first_task == "END":  #  not used at the moment
                break
            elif first_task == "SCATTER":
                data = tasks_to_accomplish.get_nowait()
                x = data.iloc[:, :1]  # vector X with samples
                y = data.iloc[:, 1]  # vector y with labels
                plt.figure(figure_num)
                plt.scatter(x, y)
                plt.title("Market Size - Training Dataset")
                plt.xlabel("Population of City in 10,000s")
                plt.ylabel("Profit in $10,000s")
                plt.show()
                tasks_that_are_done.put("Figure {} closed in Process {}".format(figure_num, current_process().name))
                time.sleep(.5)
                break

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
                tasks_that_are_done.put("Figure {} closed in Process {}".format(figure_num, current_process().name))
                time.sleep(.5)
                break

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
                tasks_that_are_done.put("Figure {} closed in Process {}".format(figure_num, current_process().name))
                time.sleep(.5)
                break

            elif first_task == "CONTOUR":
                x = tasks_to_accomplish.get_nowait()  # theta0_values
                y = tasks_to_accomplish.get_nowait()  # theta1_values
                z = tasks_to_accomplish.get_nowait()  # Cost Function J
                theta0 = tasks_to_accomplish.get_nowait()  # theta0
                theta1 = tasks_to_accomplish.get_nowait()  # theta1
                x_mesh, y_mesh = np.meshgrid(x, y)

                fig = plt.figure(figure_num)
                plt.title("Cost Function, J, Contour")
                plt.xlabel("Theta0")
                plt.ylabel("Theta1")
                plt.contour(x_mesh, y_mesh, z, 20, cmap='RdGy')

                plt.plot(theta0, theta1, 'x', color="r")
                """
                contours = plt.contour(x_mesh, y_mesh, z, 3, colors='black')
                plt.clabel(contours, inline=True, fontsize=8)

                plt.imshow(z, extent=[0, 5, 0, 5], origin='lower',
                           cmap='RdGy', alpha=0.5)
                plt.colorbar()
                """

                plt.show()
                tasks_that_are_done.put("Figure {} closed in Process {}".format(figure_num, current_process().name))
                time.sleep(.5)
                break
            else:
                continue  # go back up to the loop

        except queue.Empty:
            continue

    return True
