import numpy as np

l = 10
times = np.array([1, 5, 6]) * 0.5
step = np.arange(10*0.5)
l = step.shape[0]
step = step.reshape((1, l))
t = times.shape[0]
times = times.reshape((t, 1))
print(np.less(step, times))