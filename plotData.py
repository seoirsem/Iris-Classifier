import numpy as np
from matplotlib import pyplot as plt




def PrintErrorConvergance(e1,e2):
    # e is an array of each cost value

    plt.figure()
    plt.grid()
    m = len(e1)
    x = range(0,m)
    plt.plot(x,e1,'r+',label = 'Analytical')
    plt.plot(x,e2,'b+', label = 'Backpropogation')
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost Function')
    plt.legend()
    plt.show()
