import sys
import numpy as np
import matplotlib

print("Python", sys.version)
print("Numpy", np.__version__)
print("Matplotlib", matplotlib.__version__)

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias = [2.0,3.0,0.5]

layer_output = []
for n_weight, n_bias in zip(weights, bias) :
    neuron_output = 0
    for neu_input,neu_weight in zip(inputs,n_weight) :
        neuron_output = neuron_output+ neu_input*neu_weight
    neuron_output = neuron_output+n_bias
    layer_output.append (neuron_output)

print("layer_output from Scratch Python", layer_output)    

output = np.dot(weights,inputs)+ bias
print("layer_output using Mumpy", output)   