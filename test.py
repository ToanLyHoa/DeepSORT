import enum
from unicodedata import numeric
from DeepSort import kalman_filter as Kalman_Filter
from DeepSort import linear_assignment
import numpy as np

def temp(a,b,c,d):
    return np.array([[1,2,3,4],[5,7,6,8],[9,10,11,12]], dtype=np.float64)

min_cost_matching = linear_assignment.min_cost_matching

a, b, c = min_cost_matching(temp, 9, [1,2,3], [1,2,3,4])
print(a)
print(b)
print(c)