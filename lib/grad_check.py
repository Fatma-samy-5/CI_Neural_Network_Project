import numpy as np
from .layers import Dense
from .activations import Activation
from .losses import Loss

def check_gradients(network, X, Y, epsilon=1e-7):
    """
    Compares the analytic gradients (backpropagation) with the numerical gradients.
    """
    
    # Run forward and backward passes once to compute analytic gradients
    Y_pred = network.forward(X)
    loss = network.loss_function.loss(Y, Y_pred)
    gradient = network.loss_function.gradient(Y, Y_pred)
    network.backward(gradient)
    
    layers = [layer for layer in network.layers if isinstance(layer, Dense)]
    
    max_diffs = []
    
    print("----------------------------------------------------------------------")
    print("Layer | Weight Shape | Max Diff | Status")
    print("----------------------------------------------------------------------")

    for i, layer in enumerate(layers):
        
        # --- Check Weights (W) ---
        analytic_grad_W = layer.weights_gradient
        
        # Numerical Gradient calculation for Weights
        numerical_grad_W = np.zeros_like(layer.weights)
        original_W = layer.weights.copy()
        
        it = np.nditer(layer.weights, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            
            # f(x + epsilon)
            layer.weights[idx] = original_W[idx] + epsilon
            loss_plus = network.loss_function.loss(Y, network.forward(X))
            
            # f(x - epsilon)
            layer.weights[idx] = original_W[idx] - epsilon
            loss_minus = network.loss_function.loss(Y, network.forward(X))
            
            # Numerical gradient formula: (f(x+epsilon) - f(x-epsilon)) / (2 * epsilon)
            numerical_grad_W[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            # Restore the original weight value
            layer.weights[idx] = original_W[idx]
            
            it.iternext()
        
        # Calculate relative difference between analytic and numerical gradients
        numerator = np.abs(analytic_grad_W - numerical_grad_W)
        denominator = np.abs(analytic_grad_W) + np.abs(numerical_grad_W)
        relative_diff = numerator / denominator
        
        # To avoid division by zero when both are zero
        relative_diff[denominator == 0] = 0.0
        
        max_diff_W = np.max(relative_diff)
        max_diffs.append(max_diff_W)
        
        status_W = "OK" if max_diff_W < 1e-6 else "FAIL"
        print(f" W{i+1}   | {layer.weights.shape} | {max_diff_W:.8f} | {status_W}")
        
        # --- Check Biases (B) ---
        analytic_grad_B = layer.bias_gradient
        
        # Numerical Gradient calculation for Biases
        numerical_grad_B = np.zeros_like(layer.bias)
        original_B = layer.bias.copy()
        
        it = np.nditer(layer.bias, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            
            # f(x + epsilon)
            layer.bias[idx] = original_B[idx] + epsilon
            loss_plus = network.loss_function.loss(Y, network.forward(X))
            
            # f(x - epsilon)
            layer.bias[idx] = original_B[idx] - epsilon
            loss_minus = network.loss_function.loss(Y, network.forward(X))
            
            numerical_grad_B[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            
            layer.bias[idx] = original_B[idx]
            
            it.iternext()

        max_diff_B = np.max(np.abs((analytic_grad_B - numerical_grad_B) / (np.abs(analytic_grad_B) + np.abs(numerical_grad_B))))
        max_diffs.append(max_diff_B)
        
        status_B = "OK" if max_diff_B < 1e-6 else "FAIL"
        print(f" B{i+1}   | {layer.bias.shape} | {max_diff_B:.8f} | {status_B}")

    print("----------------------------------------------------------------------")
    return max(max_diffs)