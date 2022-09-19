import numpy as np
from sigmoid import SigmoidArray
import math

# calculates the network output using the given weights and training sample x
def CalculateOutputValue(x,thetas1,thetas2):

    # calculate hidden layer values
    z1 = np.matmul(thetas1,x)
    a1 = SigmoidArray(z1)

    # add bias term
    a1 = np.append(a1,1, axis = None)

    # calculate output value
    z2 = np.matmul(thetas2,a1)
    a2 = SigmoidArray(z2)

    return [x,a1,a2]

def CostFunction(X,ys,thetas1,thetas2):
    # calculates the cost function, where y is 1 for true and 0 for false
    # thetas1 are the first layer weights, and thetas2 are the second
    # note that each layer has a bias term of 1
    # No regularisation terms are included

    J = 0

    [p,m] = ys.shape

    for j in range(0,p): # loop over each class
        for i in range(0,m): # loop over the training data
            h = CalculateOutputValue(X[i,:],thetas1,thetas2)[2]
            if ys[j,i] == 1:
                J += -math.log(h[j])
            else:
                J += -math.log(1-h[j])
                
        J = J/m

    return J



def CreateClassifierOutputArrays(y,labels):
    #changes string outputs to three arrays for the three way classification problem

    m = len(y)

    ySetosa = np.zeros(shape = (1,m))
    yVersicolor = np.zeros(shape = (1,m))
    yVirginica = np.zeros(shape = (1,m))

    ys = np.empty(shape = (3,m))

    for i in range(0,m):
        if y[i] == labels[0]:
            ySetosa[0,i] = 1
            yVersicolor[0,i] = 0
            yVirginica[0,i] = 0
        elif y[i] == labels[1]:
            ySetosa[0,i] = 0
            yVersicolor[0,i] = 1
            yVirginica[0,i] = 0
        else:
            ySetosa[0,i] = 0
            yVersicolor[0,i] = 0
            yVirginica[0,i] = 1

    ys[0,:] = ySetosa
    ys[1,:] = yVersicolor
    ys[2,:] = yVirginica

    return ys
    
    
def AnalyticGradiant(X,ys,thetas1,thetas2,e):
   # use this code to calculate a few partial derivitaves anylitacally to check the code is working correctly
    
    [m,n] = np.shape(X)
    [t1x,t1y] = np.shape(thetas1)
    [t2x,t2y] = np.shape(thetas2)

    delta1 = np.zeros(shape = (t1x+1,t1y),dtype=float)
    #adding an extra line of zeros - error on the hidden layer bias term
    delta2 = np.zeros(shape = (t2x,t2y),dtype=float)
    
    for i in range(0,t2x):
        for j in range(0,t2y):
            t2 = thetas2
            t2[i,j] += e
            fHigh = CostFunction(X,ys,thetas1,t2)
            t2[i,j] -= 2*e
            fLow = CostFunction(X,ys,thetas1,t2)
            delta2[i,j] = (fHigh - fLow)/(2*e)

    for i in range(0,t1x):
        for j in range(0,t1y):
            t1 = thetas1
            t1[i,j] += e
            fHigh = CostFunction(X,ys,t1,thetas2)
            t1[i,j] -= 2*e
            fLow = CostFunction(X,ys,t1,thetas2)
            delta1[i,j] = (fHigh - fLow)/(2*e)
    
    return [delta1,delta2]

def Backpropogation(X,ys,thetas1,thetas2):

    [m,n] = np.shape(X)
    [t1x,t1y] = np.shape(thetas1)
    [t2x,t2y] = np.shape(thetas2)

    delta1 = np.zeros(shape = (t1x+1,t1y),dtype=float)
    #adding an extra line of zeros - error on the hidden layer bias term
    delta2 = np.zeros(shape = (t2x,t2y),dtype=float)

#    for x, y in zip(X, ys):
# make ys the correct orientation

    for i in range(0,m):
    # looping over training data
        
        # returns activations of all layers
        [x, a2, a3] = CalculateOutputValue(X[i,:],thetas1,thetas2)

        del3 = a3 - ys[:,i] #error values on output layer
        
        del2 = np.matmul(np.transpose(thetas2),del3) * (a2 * (1-a2))
    
        # Remember, no del1 term as the first layer "x" is the inputs and is assumed to have no error        
        delta2 += np.matmul(np.atleast_2d(del3).T,np.atleast_2d(a2))

        delta1 = delta1 + np.matmul(np.atleast_2d(del2).T,np.atleast_2d(x))

    delta2 = delta2/m
    delta1 = delta1/m
    
    # delta1 contains an extra row of "0" corresponding to the bias term in the hidden layer

    return [delta1, delta2]

def GradiantDescent(thetas1,thetas2,delta1,delta2,alpha):

    newTheta1 = thetas1 - alpha*delta1[0:4,:]
    # simply removing the row of "0" errors associated with the error term

    newTheta2 = thetas2 - alpha*delta2

    return[newTheta1,newTheta2]

