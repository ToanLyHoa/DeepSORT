import enum
from unicodedata import numeric
from DeepSort import kalman_filter as Kalman_Filter
from DeepSort import linear_assignment
from DeepSort import detection
import numpy as np

def temp(a,b,c,d):
    return np.array([[1,2,3,4],[5,7,6,8],[9,10,11,12]], dtype=np.float64)

a = detection.Detection([1,1,2,3], 1, 1)

print(a.to_tlbr())
print(a.to_xyah())