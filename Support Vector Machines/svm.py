import numpy as np
import random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# generate dataset
np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5], np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))
N = inputs.shape[0]
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# kernel functions
def linearKernel(x, y):
    return np.dot(x, y)

def polynomialKernel(x, y, p = 3):
    return np.power(np.dot(x, y) + 1, p)

def rbfKernel(x, y, sigma = 2):
    return math.exp(-np.dot(np.subtract(x, y), np.subtract(x, y))/(2 * sigma * sigma))

#kernel = linearKernel
#kernel = polynomialKernel
kernel = rbfKernel

# P matrix
PMatrix = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        PMatrix[i][j] = targets[i] * targets[j] * kernel(inputs[i], inputs[j])

# zero function
def zerofun(alpha):
    return np.dot(alpha, targets)

# objective function
def objective(alpha):
    return (1/2) * np.dot(alpha, np.dot(alpha, PMatrix)) - np.sum(alpha)

# start is a vector with the initial guess of the alpha vector
# N is the number of training samples
start = np.zeros(N)
C = 10000

# B is a list of pairs of the same length as the alpha vector
# upper bounds
B = [(0, C) for b in range(N)]

# lower bounds
# B = [(0, None) for b in range(N)]

# XC is used to impose other constraints besides the bounds
XC = {'type':'eq', 'fun':zerofun}

# minimize function
ret = minimize(objective, start, bounds=B, constraints=XC)
# index 'success' holds a boolean value
if (not ret['success']):
    raise ValueError('Optimizer cannot find a solution.')

# extract non-zero alpha values
alpha = ret['x']
nonzero = [(alpha[i], inputs[i], targets[i]) for i in range(N) if abs(alpha[i]) > 10e-5]

# b value
def bValue():
    bSum = 0
    for value in nonzero:
        bSum += value[0] * value[2] * kernel(value[1], nonzero[0][1])
    return bSum - nonzero[0][2]

# indicator function
def indicator(x, y, b):
    totalSum = 0
    for value in nonzero:
        totalSum += value[0] * value[2] * kernel([x, y], value[1])
    return totalSum - b

b = bValue()

# plotting
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.')
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.')

xgrid = np.linspace(-5, 5)
ygrid = np.linspace(-4, 4)
grid = np.array([[indicator(x, y, b) for x in xgrid] for y in ygrid])
plt.contour(xgrid, ygrid, grid, (-1.0,0.0,1.0), colors=('red','black','blue'), linewidths=(1,3,1))
#plt.axis('equal')
#plt.savefig('svmplot.pdf')
plt.show()
