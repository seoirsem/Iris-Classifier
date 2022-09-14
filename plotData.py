import numpy as np
from matplotlib import pyplot as plt




def PrintErrorConvergance(e):
    # e is an array of each cost value

    plt.figure()
    plt.grid()
    m = len(e)
    x = range(0,m)
    plt.plot(x,e,'r+',label = 'Analytical')
    #plt.plot(x,e2,'b+', label = 'Backpropogation')
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost Function')
    #plt.legend()
    plt.show()
