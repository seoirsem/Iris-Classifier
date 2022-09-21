import os
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from random import randint
import numpy as np
from matplotlib import pyplot as plt

# Importing some tools from previous implementation (data management etc). I also include edited versions of some functions below.
from main import PrepareInputData
from data_management import NormaliseTestData, NormaliseData
from neural_network_commands import CreateClassifierOutputArrays

class NeuralNetwork(nn.Module):
    def __init__(self):
        # number of nodes in the single hidden layer
        self.hidden = 5
        
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(4, self.hidden, bias=True),
            nn.Sigmoid(),
            nn.Linear(self.hidden, 3, bias=True),
            nn.Sigmoid(),

        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    

def InitialiseArrays(trainingData,labels):
    
    X1 = trainingData[0]
    y = trainingData[1]
    m,n = X1.shape
    X = np.zeros(shape = (m,n))
    for i in range(m):
        X[i,:] = X1[i,:].astype(float)
    
    ys1 = CreateClassifierOutputArrays(y,labels)
    
    p,m = ys1.shape
    ys = np.zeros(shape = (p,m))
    for i in range(p):
        ys[i,:] = ys1[i,:].astype(float)
    
    # Normalise the data in X (mean set to 0 and std to 1)
    # Mean and std are saved to apply to the test set later
    [X, mean, std] = NormaliseData(X)

    return X,ys.T,mean,std

def InitialiseTestArrays(trainingData,labels,mean,std):
    
    X1 = trainingData[0]
    y = trainingData[1]
    m,n = X1.shape
    X = np.zeros(shape = (m,n))
    for i in range(m):
        X[i,:] = X1[i,:].astype(float)
    
    ys1 = CreateClassifierOutputArrays(y,labels)
    
    p,m = ys1.shape
    ys = np.zeros(shape = (p,m))
    for i in range(p):
        ys[i,:] = ys1[i,:].astype(float)
    
    # Normalise the data in X (mean set to 0 and std to 1)
    # Mean and std are saved to apply to the test set later
    X = NormaliseTestData(X,mean,std)

    return X, ys.T

def RunBackpropogationOptimisation(model,X,y,epochs,learningRate):

    loss_function = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    losses = []
    start = time.time()
    for epoch in range(epochs):

        yPrediction = model(X.float())
        
        loss = loss_function(yPrediction.reshape(-1).float(), y.reshape(-1).float())
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()
        optimizer.step()
    end = time.time()
    print('Total training time: ' + str(round(end - start,2)) + 's for ' + str(epochs) + ' epochs at a learning rate of ' + str(learningRate) + '.')
    
    return model, losses

def main():

    # header    
    plotLossFunction = True
    learningRate = 0.5
    epochs = 5000
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)

    # Prepare data arrays
    strLabels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    trainingData, testData = PrepareInputData(False)
    data, labels, mean, std = InitialiseArrays(trainingData, strLabels)

    X = torch.from_numpy(data)
    y = torch.from_numpy(labels)

    # Run optimisation. Note that I don't bother to save the model as training only takes ~1s anyway
    model, losses = RunBackpropogationOptimisation(model,X,y,epochs,learningRate)


    # Look at learning
    if plotLossFunction:
        plt.figure()
        plt.plot(losses)
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Cross Entropy Loss')
        plt.show()

    # Run test data
    xTestNP, yTestNP = InitialiseTestArrays(testData,strLabels,mean,std)
    [m,n] = np.shape(xTestNP)
    xTest = torch.from_numpy(xTestNP)

    yPred = model.forward(xTest.float())

    numberCorrect = 0
    totalNumber = 0
    for pred in yPred:
        labelValue = np.argmax(yTestNP[totalNumber,:])
        totalNumber += 1
        n = float(torch.argmax(pred))
        if n==labelValue:
            numberCorrect += 1
        
    percentage = round((100.0*numberCorrect)/totalNumber,1)
    print('The model correctly classified ' + str(percentage) + '% of the test set.')


if __name__ == "__main__":
    main()