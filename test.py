import enum
from unicodedata import numeric
from DeepSort import kalman_filter as Kalman_Filter
from DeepSort import linear_assignment
from DeepSort import detection
from DeepSort import nn_matching
from DeepSort import iou_matching
import numpy as np

def temp(a,b,c,d):
    return np.array([[1,2,3,4],[5,7,6,8],[9,10,11,12]], dtype=np.float64)

a = nn_matching._nn_euclidean_distance

# a([[-1,2], [1, -3], [-1, -4]], [[2, 3], [-2, -2]])

fun = iou_matching.iou

fun(np.asarray([1,1,2,2]), np.asarray([[2,2,2,2], [4,4,3,3], [5,5,1,1]]))

dic = {}
dic.setdefault(1, []).append([1,2,3])
dic.setdefault(1, []).append([1,2,3])
print(dic)