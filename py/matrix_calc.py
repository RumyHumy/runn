from math import tanh
import numpy as np

def mseg(E, A):
    return np.vectorize(lambda x: x*2/3)(E-A)

def func(val):
    return tanh(val)

def deriv(val):
    return 1-tanh(val)**2

W = np.matrix([[+0.5,  0.0],
               [ 0.0, -0.5],
               [+1.0, -2.0]])

B = np.matrix([[+1.0],
               [-2.0],
               [-1.0]])

X = np.matrix([[-1.0],
               [+2.0]])

AX = None

Y = np.matrix([[+0.1],
               [+0.2],
               [+0.3]])

def forward():
    global AX
    AX = np.dot(W, X) + B
    return np.vectorize(func)(AX)

def backward(grad, lrate):
    global W
    global B
    # dE/dX = dE/dY mul deriv(AX)
    grad = np.multiply(grad, np.vectorize(deriv)(AX))
    # dE/dW = dY/dE * X^T
    wsGrad = np.multiply(grad, np.transpose(X))
    W -= lrate * wsGrad 
    B -= lrate * grad
    return np.dot(np.transpose(W), grad)

print(forward())
for i in range(10000):
    backward(mseg(forward(), Y), 0.5)
print(forward())
