import sys
import numpy as np
import matplotlib

print("Python", sys.version)
print("Numpy", np.__version__)
print("Matplotlib", matplotlib.__version__)

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, 1.0, 2.0],
          [-1.5, 2.7, 3.3, 0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

bias1 = [2.0,3.0,0.5]

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, 0.33],
           [-0.44, 0.73, -0.1]]

bias2 = [-1.0,2.0,-0.5]

Layer1_output = np.dot(inputs,np.array(weights1).T)+ bias1

Layer2_output = np.dot(Layer1_output,np.array(weights2).T)+ bias2
print("\n layer_output using Mumpy\n\n", Layer2_output)   