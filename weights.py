import json, codecs
import numpy as np
from matplotlib import pyplot as plt


from main import PrepareInputData, InitialiseArrays
from data_management import NormaliseTestData
from neural_network_commands import CreateClassifierOutputArrays
from sigmoid import SigmoidArray

### goal of this is to post-hoc analyse how the network functions by looking at the weights

def LoadWeightData():
    filePath1 = "weights1.json"
    filePath2 = "weights2.json"

    obj_text = codecs.open(filePath1, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    thetas1Array = np.array(b_new)

    obj_text = codecs.open(filePath2, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    thetas2Array = np.array(b_new)

    nStep,m,n = thetas1Array.shape
    nStep,o,p = thetas2Array.shape
    # these arrays contain the weights from the last training iteration


    print('Data from ' + str(nStep - 1) + ' iterations.')
    # -1 as you actually take the 0th and the end training iterations


    thetas1 = thetas1Array[-1,:,:]
    thetas2 = thetas2Array[-1,:,:]
    # these arrays contain the weights from the last training iteration
    return thetas1, thetas2, thetas1Array, thetas2Array

def RunSingleValue(x,thetas1,thetas2):
    

    z1 = np.matmul(thetas1,x)
    a1 = SigmoidArray(z1)

    # add bias term
    a1 = np.append(a1,1, axis = None)

    # calculate output value
    z2 = np.matmul(thetas2,a1)
    a2 = SigmoidArray(z2)
 
    return a1,a2,z1,z2


def CalculateNodeAndImportanceValues(X,n,thetas1,thetas2):
    def CalculateNodeImportance(x,theta):
        lenx = len(x)
        out = np.zeros(lenx)
        for i in range(lenx):
            out[i] = x[i]*sum(theta[:,i])

        return out
    def CalculateNodeImportanceAbs(x,theta):
        lenx = len(x)
        out = np.zeros(lenx)
        for i in range(lenx):
            out[i] = abs(x[i])*sum(map(abs,theta[:,i]))

        return out
    
    x = X[n,:]
    a1,a2,z1,z2 = RunSingleValue(x,thetas1,thetas2)
    xImp = CalculateNodeImportance(x,thetas1)
    a1Imp = CalculateNodeImportance(a1,thetas2)
    xImpAbs = CalculateNodeImportanceAbs(x,thetas1)
    a1ImpAbs = CalculateNodeImportanceAbs(a1,thetas2)


    return a1,a2,z1,z2,xImp,a1Imp,xImpAbs,a1ImpAbs

class ImportanceArray():

    def __init__(self,dataLabel,x,z1,z2,a1,a2,xImp,a1Imp,flower,xImpAbs,a1ImpAbs):
        self.flower = flower
        self.x = x
        self.z1 = z1
        self.z2 = z2
        self.a1 = a1
        self.a2 = a2
        self.xImp = xImp
        self.xImpAbs = xImpAbs
        self.a1Imp = a1Imp
        self.a1ImpAbs = a1ImpAbs
        self.dataLabel = dataLabel


    def PrintProperties(self):
        z1Round = [round(x,1) for x in self.z1]
        z2Round = [round(x,1) for x in self.z2]
        xImpRound = [round(x,1) for x in self.xImp]
        a1ImpRound = [round(x,1) for x in self.a1Imp]
        xImpRoundAbs = [round(x,1) for x in self.xImpAbs]
        a1ImpRoundAbs = [round(x,1) for x in self.a1ImpAbs]
        print('The ground truth is ' + self.flower + ' or ' + str(self.dataLabel))
        print('z1: ' + str(z1Round) + '\nz2: ' + str(z2Round))
        print('Input importance: ' + str(xImpRound) + '\nHidden layer importance: ' + str(a1ImpRound))
        print('Input absolute importance: ' + str(xImpRoundAbs) + '\nHidden layer importance: ' + str(a1ImpRoundAbs))


def main():
    trainingData, testData = PrepareInputData(False)
    thetas1, thetas2, thetas1Array, thetas2Array = LoadWeightData()
    # first two are the final thetas, the second pair include every theta value
    # every theta value is logged in case I want to examine how the weights are learnt at a future date
    labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    X,ys,mean,std,randomTheta1,randomTheta2 = InitialiseArrays(trainingData,labels)

    X = testData[0].copy()
    [m,n] = np.shape(X)
    X = NormaliseTestData(X,mean,std)
    X = np.append(X,np.ones(shape = (m,1)), axis = 1)
    dataLabels = CreateClassifierOutputArrays(testData[1],labels)
    printNames = testData[1]

    # calculates node based outputs    
    data = []
    for n in range(m):
        a1,a2,z1,z2,xImp,a1Imp,xImpAbs,a1ImpAbs = CalculateNodeAndImportanceValues(X,n,thetas1,thetas2)
        data.append(ImportanceArray(dataLabels[:,n],X[n,:],z1,z2,a1,a2,xImp,a1Imp,printNames[n],xImpAbs,a1ImpAbs))

    # prints a sample of the data and the weights
    data[5].PrintProperties()
    data[16].PrintProperties()
    data[30].PrintProperties()

    print(thetas1)
    print(thetas2)

if __name__ == "__main__":
    main()
