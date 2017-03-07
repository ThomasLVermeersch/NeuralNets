import numpy as np

#Sigmoid function
def nonlinear(x, derivative=False):
    if(derivative):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

#Input Data

x = np.array([
        [10, 8, 1],
        [1, 1, 1],
        [1, 10, 5],
        [5, 5, 5]
        ])
y = np.array([
        [1],
        [1],
        [0],
        [0]
        ])
np.random.seed(1)

#Synapses
synapse0 = 2 * np.random.random((3,4)) - 1
synapse1 = 2 * np.random.random((4,1)) - 1 
                               
#Training
for j in range(50000):
    layer0 = x
    layer1 = nonlinear(np.dot(layer0, synapse0))
    layer2 = nonlinear(np.dot(layer1, synapse1))
    
    layer2Error = y - layer2
    
    layer2Delta = layer2Error * nonlinear(layer2, derivative=True)
    layer1Error = layer2Delta.dot(synapse1.T)
    layer1Delta = layer1Error * nonlinear(layer1, derivative=True)
    
    #Update synapse weights
    synapse0 += layer0.T.dot(layer1Delta)
    synapse1 += layer1.T.dot(layer2Delta)

print("Output")
print(layer2)