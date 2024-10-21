# 🤖 Building Your Deep Neural Network: Step by Step

Welcome to the repository for "Building Your Deep Neural Network: Step by Step"! This project demonstrates the implementation of deep learning concepts, specifically focusing on the architecture of neural networks. 

## 📦 Packages

To get started, we need to import the necessary libraries:

```python
import numpy as np  # For numerical operations and matrix manipulations
import h5py  # For working with HDF5 file formats
import matplotlib.pyplot as plt  # For data visualization
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward  # Activation functions and their gradients

# Set random seed for reproducibility
np.random.seed(1)

# Configure Matplotlib parameters for plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)  # Default plot size
plt.rcParams['image.interpolation'] = 'nearest'  # No interpolation for image display
plt.rcParams['image.cmap'] = 'gray'  # Set default color map to grayscale

# Test case import
from testCases_v4a import *  # Importing test cases for validation of neural network models
```

## 🔧 Initialization

### 2-Layer Neural Network

Here we initialize the parameters of a two-layer neural network.

```python
def initialize_parameters(n_x, n_h, n_y):
    """
    Initialize the parameters of a two-layer neural network.
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # Sanity check
    assert W1.shape == (n_h, n_x), "Shape mismatch in W1"
    assert b1.shape == (n_h, 1), "Shape mismatch in b1"
    assert W2.shape == (n_y, n_h), "Shape mismatch in W2"
    assert b2.shape == (n_y, 1), "Shape mismatch in b2"

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Initialize parameters
parameters = initialize_parameters(3, 2, 1)
print("Initialized Parameters:")
print("W1 =", parameters["W1"])
print("b1 =", parameters["b1"])
print("W2 =", parameters["W2"])
print("b2 =", parameters["b2"])
```

### L-Layer Neural Network

We also provide a function to initialize parameters for a deep neural network with multiple layers.

```python
def initialize_parameters_deep(layer_dims):
    """
    Initialize parameters for an L-layer deep neural network.
    """
    parameters = {}
    L = len(layer_dims)  # Number of layers in the network

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))
        
        # Ensure shapes are correct
        assert parameters[f'W{l}'].shape == (layer_dims[l], layer_dims[l-1]), f"Shape mismatch in W{l}"
        assert parameters[f'b{l}'].shape == (layer_dims[l], 1), f"Shape mismatch in b{l}"

    return parameters

# Example initialization for a deep network
parameters = initialize_parameters_deep([5, 4, 3])
print("Deep Network Parameters:")
print("W1 =", parameters["W1"])
print("b1 =", parameters["b1"])
print("W2 =", parameters["W2"])
print("b2 =", parameters["b2"])
```

## 🔄 Forward Propagation

The forward propagation module includes methods for calculating linear activation and the L-layer model.

```python
def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    """
    Z = np.dot(W, A) + b
    assert Z.shape == (W.shape[0], A.shape[1]), "Shape mismatch in Z"
    return Z, (A, W, b)

# Add more forward propagation implementations as needed...
```

## 💡 Cost Function

The cost function for evaluating model performance.

```python
def compute_cost(AL, Y):
    """
    Compute the cross-entropy cost for binary classification.
    """
    m = Y.shape[1]
    cost = (-1 / m) * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
    cost = np.squeeze(cost)
    assert cost.ndim == 0, "Cost should be a scalar."
    return cost
```

## 🔙 Backward Propagation

Backward propagation for optimizing the parameters.

```python
def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer.
    """
    A_prev, W, _ = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert dA_prev.shape == A_prev.shape, "Shape of dA_prev is incorrect"
    return dA_prev, dW, db
```

## 🎓 Acknowledgments

This project is inspired by the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) from DeepLearning.AI. A huge thank you to Andrew Ng and the team for providing such valuable resources for learning deep learning!

## 🚀 Getting Started

To run this code, clone this repository and ensure you have the required packages installed. Then, follow the provided examples to understand how to implement a deep neural network from scratch!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
