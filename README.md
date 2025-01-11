# Neural-Network-Implementation

This project demonstrates the implementation of a basic neural network from scratch using only Python and `numpy`. It is designed for binary classification problems, where the network is trained to predict an output (either 0 or 1) based on given input features.

## Project Overview

The neural network consists of:
- **Input Layer**: Accepts the input features.
- **Hidden Layer**: Performs the activation and passes information to the next layer.
- **Output Layer**: Provides the final prediction.

In this implementation:
- A **feedforward neural network** with one hidden layer is created.
- **Sigmoid activation function** is used for both the hidden and output layers.
- **Mean squared error** is used for error calculation.
- **Gradient descent** is used for training the network by updating the weights and biases.

### Key Features
- Custom manual implementation of a neural network.
- Binary classification using a simple dataset.
- Use of backpropagation for weight updates.
- Training with specified epochs to minimize the error.


## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/Numairazaib19/Neural-Network-Implementation.git
   ```

2. Navigate into the project directory:

   ```bash
   cd Neural-Network-Implementation
   ```

3. Run the code:

   ```bash
   python main.py
   ```

The neural network will train over a specified number of epochs (in this case, 3) and print the predicted outputs along with the error at each epoch.


## How It Works

- **Forward Propagation**: The input is passed through the network layers, and at each layer, the weighted sum of inputs is calculated and passed through the sigmoid activation function.
  
- **Backpropagation**: After calculating the output, the error is computed (target - predicted output). This error is propagated back through the network to adjust the weights and biases using the gradient descent method.

- **Training**: The network is trained for a fixed number of epochs, where in each epoch, weights and biases are updated based on the gradients calculated during backpropagation.

