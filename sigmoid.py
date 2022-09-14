
from math import exp

import numpy as np


def SigmoidArray(A):
    # operates sigmoid function on each (numpy) array element

    def Sigmoid(x):
        return 1 / (1 + exp(-x))

    ArrayFunction = np.vectorize(Sigmoid)
    
    return ArrayFunction(A)
