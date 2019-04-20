"""
Simplistic implementation of the two-layer neural network.
Training method is stochastic (online) gradient descent with momentum.
As an example it computes XOR for given input.
Some details:
- tanh activation for hidden layer
- sigmoid activation for output layer
- cross-entropy loss
Less than 100 lines of active code.
"""

import numpy as np
import time

#variables
n_hidden = 1
n_in = 1
#outputs
n_out = 1
#sample
n_samples = 1
#________________________________________________________________________________________
#Hyper Parameters
learning_rate = 0.01
momentum = 0.9
#________________________________________________________________________________________
#non determinisitc Seeding
np.random.seed(0)
#________________________________________________________________________________________
#Activation Functin one
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
#________________________________________________________________________________________
#Activation Function Two
def tanh_prime(x):
    return  1 - np.tanh(x)**2
#________________________________________________________________________________________
#Training Function
# x=Input data; t=Transpose (Help function matrix multiplication ) ;V= Layer one; W= layer two; bv= bias one; bw= bias two
def train(x, t, V, W, bv, bw):
    # forward - martix multiply + bias is
    A = np.dot(x, V) + bv
    Z = np.tanh(A)
    print("The value in first step..Z==",Z)
    #layer two Forward prop
    B = np.dot(Z, W) + bw
    Y = sigmoid(B)
    # backward
    print("The value in Second  step..Y==", Y)
    Ew = Y - t
    print("The value in backward step one EW....==",Ew)
    #we are getting our detlas
    Ev = tanh_prime(A) * np.dot(W, Ew)
    print("The value in backward step Two Ev ....==", Ev)
    #predit our loss
    #Two detlas .. We are using here .. Z values in forwad step and the x values
    dW = np.outer(Z, Ew)
    print("The outer layer that is cross prodcuct of the first part ....==", dW)
    dV = np.outer(x, Ev)
    print("The outer layer that is cross prodcuct of the Second  part ....==", dV)
    #calicuating our loss in this scenario
    #cross entropy
    loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )
    print("The loss value  ....==",loss)
    # Note that we use error for each layer as a gradient
    # for biases
    return  loss, (dV, dW, Ev, Ew)
#________________________________________________________________________________________
def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    #what ever we retrun is going to be our prediciton
    return (sigmoid(B) > 0.5).astype(int)
#________________________________________________________________________________________
# Setup initial parameters
# Note that initialization is cruxial for first-order methods!
#layer one
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
#layer two
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))
#bias one , using the hidden layer
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)
#imput paramenter .. arry of values
params = [V,W,bv,bw]
# Generate some data
#we are generating 300 records
X = np.random.binomial(1, 0.5, (n_samples, n_in))
T = X ^ 1
#________________________________________________________________________________________
# Train
for epoch in range(100):
    #error array
    err = []
    #update varialbe .. equal to len of pramenters initialized on the top
    upd = [0]*len(params)
    print("The upd value in up loop one  ....==", upd)
    #how long you want to run the nural net
    t0 = time.clock()
    #for each data point update our weight s
    #using shape .. we say the size of data
    for i in range(X.shape[0]):
        #loss and gradient using training fun that we initalized declared and created . loss and grad
        loss, grad = train(X[i], T[i], *params)
        #update our loass
        #Calucualate our loss
        print("The upd value in up loop two  loss .... grad  ....==", loss,grad)
        for j in range(len(params)):
            params[j] -= upd[j]
            print("The params  value in up loop   three( one  ).... params   ....==",params[j])
        #The hper parameters are used over here
        for j in range(len(params)):
            upd[j] = learning_rate * grad[j] + momentum * upd[j]
            print("The params  value in upidn   three( two  ).... upid   ....==", upd[j])
        #append our error with loss
        err.append( loss )
        print("Seperator __________________________________________________________________________________")
    #printing out our results
    print("Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 ))
#________________________________________________________________________________________
# Try to predict something

x = np.random.binomial(1, 0.5, n_in)
print("XOR prediction:")
print(x)
print(predict(x, *params))
