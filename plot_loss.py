import numpy as np
import matplotlib.pyplot as plt

log_path = '/Users/huangege/JDProjects/NeuralBoy/loss_log'

f = open(log_path, 'r')

lines = f.readlines()
arr = [i.split(',') for i in lines]
total_cols = len(arr[0])

x = range(len(lines))

for i in range(total_cols):
    y = [float(k[i]) for k in arr]
    plt.plot(x, np.asarray(y), label = i)

plt.xlabel('steps')
plt.ylabel('loss_value')
plt.legend()

plt.show()

