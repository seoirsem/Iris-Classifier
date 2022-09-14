import numpy as np
from neural_network_commands import CalculateOutputValue, CostFunction, CreateClassifierOutputArrays
from data_management import NormaliseTestData

def RunTestSet(testData,thetas1,thetas2,labels,mean,std):

    X = testData[0]
    [m,n] = np.shape(X)
    X = NormaliseTestData(X,mean,std)


    X = np.append(X,np.ones(shape = (m,1)), axis = 1)

    dataLabels = CreateClassifierOutputArrays(testData[1],labels)
    print('The loss on the test dataset is: ' + str(round(CostFunction(X,dataLabels,thetas1,thetas2),5)) + '.')

    networkLabels = []# = np.chararray(shape = (m,1))

    for i in range(m):
        y = CalculateOutputValue(X[i,:],thetas1,thetas2)[2]
        maxInY = max(y)
        if maxInY>0.5: # only output if max output > 0.5
            networkLabels.append(labels[np.argmax(y)])
        else:
            networkLabels.append('No clear classification')
        print('Prediction: ' + networkLabels[i] + ', Reality: ' + testData[1][i])

