import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    return 1 / (1 + np.exp(-z))


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.

    Layer 1:
        h1 = sigmoid(XW1)

    Layer 2:
        h2 = sigmoid(h1W2)

    Output layer:
        y = sigmoid(h2W3)

    Returns:
        h1, h2, y
    """
    h1 = sigmoid(np.dot(X, W1))
    h2 = sigmoid(np.dot(h1, W2))
    y = sigmoid(np.dot(h2, W3))

    return h1, h2, y


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.

    Returns:
        dW1, dW2, dW3, loss
    """

    # ----- LOSS (Binary Cross Entropy) -----
    loss = -np.mean(label * np.log(y + 1e-8) + (1 - label) * np.log(1 - y + 1e-8))

    # ----- OUTPUT LAYER GRADIENT -----
    dy = y - label  # derivative of loss wrt output (sigmoid + BCE)

    dW3 = np.dot(h2.T, dy)

    # ----- HIDDEN LAYER 2 -----
    dh2 = np.dot(dy, W3.T)
    dh2_raw = dh2 * h2 * (1 - h2)  # sigmoid derivative

    dW2 = np.dot(h1.T, dh2_raw)

    # ----- HIDDEN LAYER 1 -----
    dh1 = np.dot(dh2_raw, W2.T)
    dh1_raw = dh1 * h1 * (1 - h1)

    dW1 = np.dot(X.T, dh1_raw)

    return dW1, dW2, dW3, loss
