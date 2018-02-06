'''
Note:
Neural nets are composite function
'''


# matrix math
import numpy as np

# input data
input_data = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1,1]])
output_labels = np.array([[0, 1, 1, 0]]).T

print(input_data)
# print(output_labels)

# sigmoid function
def activate(z, deriv=False):
  if(deriv == True):
    return z*(1 - z)  # sigmoid curve gradient
  return 1/(1 + np.exp(-z))

# 2 weight values
synaptic_weights_0 = 2 * np.random.random((3, 4)) - 1
synaptic_weights_1 = 2 * np.random.random((4, 1)) - 1
print(synaptic_weights_0)
print(synaptic_weights_1)

# training
for j in range(60000):

  # Forward propagate through layer 0, 1 and 2
  layer0 = input_data
  layer1 = activate(np.dot(layer0, synaptic_weights_0))
  layer2 = activate(np.dot(layer1, synaptic_weights_1))

  # Backpropagation

  # calculate error for layer 2
  layer2_error = output_labels - layer2

  if(j % 10000) == 0:
    print("Error: ", np.mean(np.abs(layer2_error)))

  # Use it to compute the gradient
  layer2_gradient = layer2_error * activate(layer2, deriv=True)

  #calculate error for layer 1
  layer1_error = layer2_gradient.dot(synaptic_weights_1.T)

  # use it to compute its gradient
  layer1_gradient = layer1_error * activate(layer1, deriv=True)

  # update the weights using the gradients
  synaptic_weights_1 += layer1.T.dot(layer2_gradient)
  synaptic_weights_0 += layer0.T.dot(layer1_gradient)

# # testing
test_layer1 = activate(np.dot(np.array([1, 0, 0]), synaptic_weights_0))
test_layer2 = activate(np.dot(test_layer1, synaptic_weights_1))
print(test_layer2)


