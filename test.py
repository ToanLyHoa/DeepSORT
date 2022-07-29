from DeepSort import kalman_filter as Kalman_Filter
import numpy as np


kalman_filter = Kalman_Filter.KalmanFilter()

a = np.eye(2 * 4, 1)
for i in range(8):
    a[i] = i

print(kalman_filter._motion_mat)
print(kalman_filter._motion_mat * a)
print(np.matmul(kalman_filter._motion_mat, a))