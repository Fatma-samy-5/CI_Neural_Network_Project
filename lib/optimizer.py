import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, network):
        
        for layer in network.layers:
            
            if hasattr(layer, 'weights') and hasattr(layer, 'weights_gradient'):
                
                
                layer.weights -= self.learning_rate * layer.weights_gradient
                
                
                layer.bias -= self.learning_rate * layer.bias_gradient