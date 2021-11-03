import multiprocessing as mp
from ex1.run_ex1 import run_ex1
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue  # imported for using queue.Empty exception
import pandas as pd
from plot_data import *

def run_ex1_test(queue):
    # ==================== Part 1: Basic Function ====================

    print("Starting exercise 1 Test.\n")

    # ======================= Part 2: Plotting =======================
    """
    col_names = ['x0', 'y']
    data = pd.read_csv('datasets/ex1data1.txt', sep=",", names=col_names, header=None)
    # print(data)

    x = data.iloc[:, :1]  # vector X with samples
    y = data.iloc[:, 1]  # vector y with labels
    """
    queue.put("TASK TEST")
    time.sleep(.5)
    return True


def main():
    plotting_tasks = Queue()  # create queue to send data
    tasks_that_are_done = Queue()
    processes = []

    for i in range(1):
        plotting_tasks.put("Task no " + str(i))

    # creating processes
    p1 = Process(target=do_job, args=(plotting_tasks, tasks_that_are_done, 1))
    processes.append(p1)
    p1.start()

    run_ex1_test(plotting_tasks)  # put new data in queue

    p2 = Process(target=do_job, args=(plotting_tasks, tasks_that_are_done, 2))
    processes.append(p2)
    p2.start()

    # completing process
    for p in processes:
        plotting_tasks.put("END")
        time.sleep(2)

    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print("In Main: {}".format(tasks_that_are_done.get()))

    print("Program finished. Goodbye!")

    return True




if __name__ == '__main__':
    main()

