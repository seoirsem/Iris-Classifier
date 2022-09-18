import numpy as np
from matplotlib import pyplot as plt
import math


def ViewData(X,y):
    # reads the data and displays scatter charts of the variants coloured by classification
    # Note that a legend is not added for legibility
    # X is: sepal_length, sepal_width, petal_length, petal_width
    # y is: Iris-setosa, Iris-versicolor, Iris-virginica

    def plotScatter(n1,n2,X,y):
        #plots a scatter of the given two arrays variables in X and the corresponding type y. 0 <= n1,n2 < 4
        names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        plt.xlabel(names[n1] + ' (cm)')
        plt.ylabel(names[n2] + ' (cm)')
        plt.grid()

        m = len(y)

        for i in range(0,m):
            if y[i] == 'Iris-setosa':
                plt.plot(X[i,n1], X[i,n2], 'b+')
            elif y[i] == 'Iris-versicolor':
                plt.plot(X[i,n1], X[i,n2], 'g+')
            elif y[i] == 'Iris-virginica':
                plt.plot(X[i,n1], X[i,n2], color = 'orange', marker = '+')
            else:
                print('Flower type at position ' + str(i) + ' is not recognised.')

    fig = plt.figure()
    
    plt.subplot(3, 2, 1)
    plotScatter(0,1,X,y)

    plt.subplot(3, 2, 2)
    plotScatter(0,2,X,y)

    plt.subplot(3, 2, 3)
    plotScatter(0,3,X,y)

    plt.subplot(3, 2, 4)
    plotScatter(1,2,X,y)
    
    plt.subplot(3, 2, 5)
    plotScatter(1,3,X,y)
    
    ax = plt.subplot(3, 2, 6)
    plotScatter(2,3,X,y)
    
    plt.tight_layout() #needed to prevent the subplots overlapping
    plt.show()


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

def plotConvergenceData(eArray,alphas):

    plt.figure()
    plt.grid()
    m,n = eArray.shape
    x = range(n)
    for i in range(m):
        alpha = alphas[i]
        plt.plot(x,eArray[i,:],label = str(round(alpha,2)))
    plt.xlabel('Iteration Number')
    plt.ylim([0,1])
    plt.ylabel('Cost Function')
    plt.legend()
    plt.show()


