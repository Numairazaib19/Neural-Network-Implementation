import numpy as np


# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)


# Neural Network class definition
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.9):
        # Initialize parameters (weights and biases)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Use provided specific weight values (instead of random initialization)
        self.weights_input_hidden = np.array([[0.1, 0.4], [0.8, 0.6]])  # Weights from input to hidden layer
        self.weights_hidden_output = np.array([[0.3], [0.9]])  # Weights from hidden to output layer
        
        # Bias initialization (using bias values of 1.0)
        self.bias_hidden = np.ones(self.hidden_size)  # Bias for hidden layer
        self.bias_output = np.ones(self.output_size)  # Bias for output layer

    def forward(self, x):
        # Forward propagation
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden  # Weighted sum of inputs to hidden layer
        self.hidden_output = sigmoid(self.hidden_input)  # Activation function applied to hidden layer
        
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output  # Weighted sum to output layer
        self.output_output = sigmoid(self.output_input)  # Activation function applied to output layer

        return self.output_output

    def backward(self, x, y):
        # Backpropagation (Compute gradients and update weights)
        output_error = self.output_output - y  # Error in output layer
        output_delta = output_error * sigmoid_derivative(self.output_output)  # Delta for output layer
        
        # Error in hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)  # Delta for hidden layer

        # Update weights
        self.weights_hidden_output -= self.learning_rate * self.hidden_output.T.dot(output_delta)  # Update hidden-output weights
        self.weights_input_hidden -= self.learning_rate * x.T.dot(hidden_delta)  # Update input-hidden weights

        # Update biases
        self.bias_output -= self.learning_rate * np.sum(output_delta, axis=0)  # Update output bias
        self.bias_hidden -= self.learning_rate * np.sum(hidden_delta, axis=0)  # Update hidden bias

    def train(self, x, y, epochs=3):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(x)
            
            # Compute error 
            error = (y - output)
            
            # Backward pass (gradient descent)
            self.backward(x, y)

            # Print progress for each epoch
            print(f"Epoch {epoch + 1}:")
            print(f"  Predicted Output: {output[0][0]:.4f}")
            print(f"  Actual Target: {y[0]}")
            print(f"  Error: {error[0][0]:.6f}\n")


# Define training data (inputs and targets)
x_train = np.array([[0.35, 0.9]])  # Input features (x1, x2)
y_train = np.array([[1]])  # Target output

# Initialize the neural network
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.9)

# Train the neural network
nn.train(x_train, y_train, epochs=3)
