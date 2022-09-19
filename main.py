
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from os.path import isfile
import time

import json, codecs



from data_management import *
from neural_network_commands import *
from plotData import PlotErrorConvergance, PlotErrorAndPercentageCorrect, PlotErrorConverganceComparison, plotConvergenceData, ViewData
from run_test_set import RunTestSet, NumberOfEachFlower


def PrepareInputData(displayInputData):
    
    #Either reads in a test and training dataset randomly sampled from the file, or makes one if they don't already exist
    if (not isfile("TrainingData.csv")) or (not isfile("TestData.csv")):
        # extracts X - 4x150 data, and y - 1x150 classifications
        print('Creating a training and test dataset.')
        filename = "IRIS.csv"
        data = ImportData(filename)
        # ViewData(data[0],data[1])
        ProduceRandomSubsets(data[0],data[1])
    
    trainingData = ImportData("TrainingData.csv")
    testData = ImportData("TestData.csv")

    # use the following commands to view the test and training datasets:
    if displayInputData:
    #    ViewData(testData[0],testData[1])
        ViewData(trainingData[0],trainingData[1])

    return trainingData,testData


def InitialiseArrays(trainingData,labels):
    
    X = trainingData[0]
    y = trainingData[1]
    ys = CreateClassifierOutputArrays(y,labels)
    [m,n] = X.shape
    [p,m] = ys.shape
    
    # Normalise the data in X (mean set to 0 and std to 1)
    # Mean and std are saved to apply to the test set later
    [X, mean, std] = NormaliseData(X)

    X = np.append(X,np.ones(shape = (m,1)), axis = 1)
    e = 0.001 # random range of values

    # layer 1 bias variables - with a bias term
    thetas1 = np.random.uniform(low = -e, high = e, size = (n,n+1))
    # layer 2 bias variables - with a bias term
    thetas2 = np.random.uniform(low = -e, high = e, size = (p,n+1))

    #print(CostFunction(X,ys,thetas1,thetas2))
    return X,ys,mean,std,thetas1,thetas2

def RunCoreAlgorithmBack(X,ys,alpha,nStep,nSample,thetas1,thetas2):
    e = []
    numberIncorrect = []
    numberCorrect = []

    start = time.time()
    for i in range(nStep):
        e.append(CostFunction(X,ys,thetas1,thetas2))
        [delta1,delta2] = Backpropogation(X,ys,thetas1,thetas2)
        [thetas1,thetas2] = GradiantDescent(thetas1,thetas2,delta1,delta2,alpha)
        if i % nSample == 0:
            print('Step ' + str(i) + ' of ' + str(nStep))
    end = time.time()
    print('Total time: ' + str(round(end - start,2)) + 's')
    return thetas1,thetas2,e

def RunCoreAlgorithmBackSaveSteps(X,ys,testData,labels,mean,std,alpha,nStep,nSample,thetas1,thetas2):
    e = []
    numberIncorrect = []
    numberCorrect = []
    m,n = thetas1.shape
    o,p = thetas2.shape
    thetas1Array = np.empty(shape = (nStep,m,n))
    thetas2Array = np.empty(shape = (nStep,o,p))

    start = time.time()
    for i in range(nStep):
        e.append(CostFunction(X,ys,thetas1,thetas2))
        [delta1,delta2] = Backpropogation(X,ys,thetas1,thetas2)
        [thetas1,thetas2] = GradiantDescent(thetas1,thetas2,delta1,delta2,alpha)
        thetas1Array[i,:,:] = thetas1
        thetas2Array[i,:,:] = thetas2
        if i % nSample == 0:
            print(i)
            nC, nI = RunTestSet(testData,thetas1,thetas2,labels,mean,std)
            numberCorrect.append(nC)
            numberIncorrect.append(nI)
    end = time.time()
    print('Total time: ' + str(round(end - start,2)) + 's')
    return thetas1,thetas2,e,numberCorrect,numberIncorrect,thetas1Array,thetas2Array


def RunCoreAlgorithmNumerical(X,ys,alpha,nStep,nSample,thetas1,thetas2):
    eAnalytical = []

    thetasAna1 = thetas1
    thetasAna2 = thetas2
    start = time.time()
    for i in range(nStep):
        eAnalytical.append(CostFunction(X,ys,thetasAna1,thetasAna2))
        [delAnaly1,delAnaly2] = AnalyticGradiant(X,ys,thetasAna1,thetasAna2,0.0001)
        [thetasAna1,thetasAna2] = GradiantDescent(thetasAna1,thetasAna2,delAnaly1,delAnaly2,alpha)

        if i % nSample == 0:
            print(i)

    end = time.time()
    print('Total time: ' + str(round(end - start,2)) + 's')

    return thetasAna1,thetasAna1,eAnalytical



def main():
    displayInputData = False
    e = 0.001
    # this is the magnitude range of the initial seeding of the weights matrices
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    alpha = 4
    # Gradient descent step size
    nStep = 300
    # Number of iterations
    nSample = 5
    # How often we sample the output accuracy
    # How many of each flower are present in the (ground truth) of the test data
    backpropGradiantDescent = True
    saveIntermediateSteps = False
    # algorithm is slower, but tests the test set every nStep of the training to measure functional convergence
    numericalGradiantDescent = False
    outputConvergencePlot = True
    loopOverAlpha = False

    trainingData, testData = PrepareInputData(displayInputData)
    
    # here we train the network
    if loopOverAlpha:
        nAlpha = 7
        alphas = np.logspace(-0.5,1,num = nAlpha,base = 10)
        alphas = np.linspace(2,7,num = nAlpha)
        eArray = np.empty(shape = (nAlpha,nStep))
    else:
        alphas = [alpha]

    for i in range(len(alphas)):
        X, ys, mean, std, thetas1, thetas2 = InitialiseArrays(trainingData,labels)
        
        m,n = thetas1.shape
        o,p = thetas2.shape
        thetas1Array = np.empty(shape = (nStep+1,m,n))
        thetas2Array = np.empty(shape = (nStep+1,o,p))
        thetas1Array[0,:,:] = thetas1
        thetas2Array[0,:,:] = thetas2

        alpha = alphas[i]

        print(str(nStep) + ' steps at a learning rate of ' + str(round(alpha,2)))
        if backpropGradiantDescent:
            if saveIntermediateSteps:
                thetas1,thetas2,e,numberCorrect,numberIncorrect,thetas1Arr,thetas2Arr = RunCoreAlgorithmBackSaveSteps(X,ys,testData,labels,mean,std,alpha,nStep,nSample,thetas1,thetas2)   
            else:
                thetas1,thetas2,e = RunCoreAlgorithmBack(X,ys,alpha,nStep,nSample,thetas1,thetas2)
        if numericalGradiantDescent:
            thetasAna1,thetasAna1,eAnalytical = RunCoreAlgorithmNumerical(X,ys,alpha,nStep,nSample,thetas1,thetas2)

        print('The loss on the training dataset is: ' + str(round(CostFunction(X,ys,thetas1,thetas2),5)) + '.')
        if loopOverAlpha:
            eArray[i,:] = e
        if saveIntermediateSteps:
            thetas1Array[1:nStep+1,:,:] = thetas1Arr
            thetas2Array[1:nStep+1,:,:] = thetas2Arr
        
    if loopOverAlpha:
        plotConvergenceData(eArray,alphas)
    

    if saveIntermediateSteps:
        filePath1 = "weights1.json"
        filePath2 = "weights2.json"

        listTheta1 = thetas1Array.tolist()
        json.dump(listTheta1, codecs.open(filePath1, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) ### this saves the array in .json format
        listTheta2 = thetas2Array.tolist()
        json.dump(listTheta2, codecs.open(filePath2, 'w', encoding='utf-8'), 
          separators=(',', ':'), 
          sort_keys=True, 
          indent=4) ### this saves the array in .json format
        
    

    ########### TODO ####### Log the theta values for reuse
    ########## TODO ########### plot the incorrectly identified flowers using a red circle over the initial charts
    
    if outputConvergencePlot:
        if numericalGradiantDescent and backpropGradiantDescent:
            PlotErrorConverganceComparison(e,eAnalytical)
        if numericalGradiantDescent:
            PlotErrorConvergance(eAnalytical)
        elif backpropGradiantDescent:
            PlotErrorConvergance(e)
            
            


if __name__ == "__main__":
    main()
