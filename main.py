from cgi import test
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from os.path import isfile


from data_management import *
from neural_network_commands import *
from plotData import PrintErrorConvergance
from run_test_set import RunTestSet

def main():

    ################################

    #Either reads in a test and training dataset randomly sampled from the file, or makes one if they don't already exist
    if (not isfile("TrainingData.csv")) or (not isfile("TestData.csv")):
        # extracts X - 4x150 data, and y - 1x150 classifications
        print('Creating a training and test dataset.')
        filename = "IRIS.csv"
        data = ImportData(filename)
        # ViewData(data[0],data[1])
        [trainingData, testData] = ProduceRandomSubsets(data[0],data[1])
    else:
        trainingData = ImportData("TrainingData.csv")
        testData = ImportData("TestData.csv")

    # use the following commands to view the test and training datasets:
    #       ViewData(testData[0],testData[1])
    #       ViewData(trainingData[0],trainingData[1])

    ################################

    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    X = trainingData[0]
    y = trainingData[1]
    ys = CreateClassifierOutputArrays(y,labels)
    [m,n] = X.shape
    [p,m] = ys.shape

    #hiddenLayerSize = 8
    
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


    alpha = 1
    nStep = 500
    # Step size for gradient descent
    
    print(str(nStep) + ' steps at a learning rate of ' + str(alpha))
    
    e = []

    thetasAna1 = thetas1
    thetasAna2 = thetas2

    for i in range(nStep):
        e.append(CostFunction(X,ys,thetas1,thetas2))
        #errorsA.append(CostFunction(X,ys,thetasAna1,thetasAna2))

        [delta1,delta2] = Backpropogation(X,ys,thetas1,thetas2)
        [thetas1,thetas2] = GradiantDescent(thetas1,thetas2,delta1,delta2,alpha)

        #[delAnaly1,delAnaly2] = AnalyticGradiant(X,ys,thetasAna1,thetasAna2,0.0001)
        #[thetasAna1,thetasAna2] = GradiantDescent(thetasAna1,thetasAna2,delAnaly1,delAnaly2,alpha)

    #PrintErrorConvergance(e)

    ########### TODO ####### Log the theta values for reuse

    print('The loss on the training dataset is: ' + str(round(CostFunction(X,ys,thetas1,thetas2),5)) + '.')

    RunTestSet(testData,thetas1,thetas2,labels,mean,std)


if __name__ == "__main__":
    main()
