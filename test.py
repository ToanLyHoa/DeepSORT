from DeepSort import kalman_filter as Kalman_Filter
import numpy as np


kalman_filter = Kalman_Filter.KalmanFilter()

mean, covariance = kalman_filter.initiate((1,2,1,1))

print(mean)
print(covariance)

mean, covariance = kalman_filter.predict(mean, covariance)
print(mean)
print(covariance)
