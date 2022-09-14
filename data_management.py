from re import A
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random

def ImportData(filename):
    # reads the data and displays some scatter charts coloured by classification
    # sepal_length, sepal_width, petal_length, petal_width
    # Iris-setosa, Iris-versicolor, Iris-virginica

    data = pd.read_csv(filename).to_numpy()

    X = data[:,0:4]
    y = data[:,4]
    
    return [X,y]

def WriteToFile(X,y,filename):
    
    df = pd.DataFrame(X, columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    df['species'] = y
    
    df.to_csv(filename, index = False)    

def ProduceRandomSubsets(X,y):

    m = len(y)
    nTest = 37
    indices = np.linspace(0,m-1,m)
    testDataIndices = random.sample(list(indices), nTest)

    testDataX = np.zeros(shape = (nTest,4))
    testDataY = []
    trainingDataX = np.zeros(shape = (m - nTest,4))
    trainingDataY = []

    iTest = 0
    iTrain = 0
    for i in range (0,m):
        if i in testDataIndices:            
            testDataX[iTest,:] = X[i,:]
            testDataY.append(y[i])
            iTest += 1
        else:            
            trainingDataX[iTrain,:] = X[i,:]
            trainingDataY.append(y[i])
            iTrain += 1
    # TODO: OUTPUT THESE ARRAYS TO FILES TO BE READ IN FUTURE
    WriteToFile(testDataX,testDataY,'TestData.csv')
    WriteToFile(trainingDataX,trainingDataY,'TrainingData.csv')
    #dataTrain.to_

    return [[trainingDataX, trainingDataY],  [testDataX, testDataY]]



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
                plt.plot(X[i,n1], X[i,n2], 'r+')
            elif y[i] == 'Iris-versicolor':
                plt.plot(X[i,n1], X[i,n2], 'b+')
            elif y[i] == 'Iris-virginica':
                plt.plot(X[i,n1], X[i,n2], 'g+')
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
    
    plt.subplot(3, 2, 6)
    plotScatter(2,3,X,y)
    
    plt.tight_layout() #needed to prevent the subplots overlapping
    plt.show()

    
def NormaliseData(X):
    # normalise the input features
    [m,n] = X.shape

    meanArray = []
    stdArray = []

    for i in range(0,n):
        mean = np.mean(X[:,i])
        std = np.std(X[:,i])
        #print(mean,std)
        for j in range(0,m):
            X[j,i] = (X[j,i]-mean)/std
        
        meanArray.append(mean)
        stdArray.append(std)

    return [X, meanArray, stdArray]

def NormaliseTestData(X,mean,std):

    [m,n] = X.shape
    for i in range(0,n):

        for j in range(0,m):
            X[j,i] = (X[j,i]-mean[i])/std[i]
        

    return X
