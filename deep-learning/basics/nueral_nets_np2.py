import numpy as np

class NeuralNetwork():
  def __init__(self):
    # Seed the random number generator, so it generates the same numbers every time the program runs.
    np.random.seed(1)

    # We model a single neuron, with 3 input connection and 1 output connection.
    # We assign random weights t0 3 x 1 matrix, with values in range -1 to 1 and mean 0.
    self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

  # The sigmoid function, which describes and S shaped curve.
  # We pass the weighted sum of the inputs through this function to normalise them between 0 and 1
  def __sigmoid(self, z):
    return 1 / (1 + np.exp(-z))

  # The derivate of the sigmoid function.
  # This is the gradient of the Sigmoid curve.
  # It indicates how confident we are about the exiting weight
  def __sigmoid_derivatives(self, x):
    return x * (1 - x)

  # We train the neural network through a process of trial and error.
  # Adjusting the synaptic weights each time.
  def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
    for iteration in range(number_of_training_iterations):
      # pass the training set through our neural network(a single neuron).
      output = self.think(training_set_inputs)

      # calculate the error (The difference btw the desired and predicted output)
      error = training_set_outputs - output

      # Multiply the error by the input and again by the gradient of the Sigmoid curve.
      # This means less confident weights are adjusted more.
      # This means inputs, which are zero, do not cause changes to the weights.
      adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivatives(output))

      # Adjust the weights
      self.synaptic_weights += adjustment

  # The neural network thinks
  def think(self, inputs):
    # Pass inputs through our neural network(our single neuron)
    return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

if __name__ == "__main__":
  # Initialise a single neural network
  neural_network = NeuralNetwork()

  print("Random starting synaptic weights: ")
  print(neural_network.synaptic_weights)

  # The training set. We have 4 examples, each consisting of 3 inputs values and 1 output value
  training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
  training_set_outputs = np.array([[0, 1, 1, 0]]).T

  # Train the neural network using a training set.
  # Do it 10,000 times and make small adjustments each time.
  neural_network.train(training_set_inputs, training_set_outputs, 10000)

  print("New sysnaptic weights after training: ")
  print(neural_network.synaptic_weights)

  # Test the neural network with a new situation.
  print("Considering new situation [1, 0, 0] -> ?: ")
  print(neural_network.think(np.array([1, 0, 0])))
