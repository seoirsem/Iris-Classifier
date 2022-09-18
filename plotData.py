import numpy as np
from matplotlib import pyplot as plt
import math



def PlotErrorConvergance(e):
    # e is an array of each cost value

    plt.figure()
    plt.grid()
    m = len(e)
    x = range(m)
    plt.plot(x,e,'r+',label = 'Loss function with time')
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost Function')
    plt.show()

def PlotErrorConverganceComparison(eBackProp, eAnalytical):
    # e is an array of each cost value

    plt.figure()
    plt.grid()
    m = len(eBackProp)
    x = range(m)
    plt.plot(x,eBackProp,'r+',label = 'Backpropogation')
    plt.plot(x,eAnalytical,'b+', label = 'Analytical')
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost Function')
    plt.legend()
    plt.show()

def PlotErrorAndPercentageCorrect(e, numberCorrect, numberIncorrect,testFlowerCount,nSample,labels):
    # e is a 1xm array, number correct is an array of 1x3 arrays, number incorrect is a 1xm/nSample array
    nTest = sum(testFlowerCount)
    m = len(e)

    xe = range(m)
    xS = range(0,m,nSample)
    n = math.floor(m/nSample)

    #print(xS)
    #print(n)

    proportionCorrect = []
    numberEachCorrect = np.zeros(shape = (3,n))
    for i in range(len(numberCorrect)):
        proportionCorrect.append(sum(numberCorrect[i])/nTest)
        numberEachCorrect[:,i] = np.divide(numberCorrect[i],testFlowerCount)
    #print(numberEachCorrect[0,:])

    fig,ax = plt.subplots()
    ax.grid()
    ax.plot(xe,e,'r+',label = 'Loss function with time')
    ax2=ax.twinx()
    ax2.plot(xS,proportionCorrect, label = 'Proportion of flowers correctly classified')
    #ax2.plot(xS,numberEachCorrect[0,:], label = labels[0])
    #ax2.plot(xS,numberEachCorrect[1,:], label = labels[1])
    #ax2.plot(xS,numberEachCorrect[2,:], label = labels[2])
    
    plt.xlabel('Iteration Number')
    ax.set_ylabel('Cost Function')
    ax2.set_ylabel('Proportion correctly classified')
    #plt.legend()
    plt.show()
