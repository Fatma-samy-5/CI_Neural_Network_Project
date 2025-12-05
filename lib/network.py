import numpy as np

from .layers import Layer, Dense
from .activations import Activation
from .losses import Loss, MeanSquaredError

class Network:
    def __init__(self):
        self.layers = []
        self.loss_function = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_function = loss

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, output_gradient):
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)

    def train_step(self, x_batch, y_batch):
        
        y_pred = self.forward(x_batch)

        loss = self.loss_function.loss(y_batch, y_pred)
        gradient = self.loss_function.gradient(y_batch, y_pred)

        self.backward(gradient)
        
        # التعديل: استدعاء دالة update في المُحسّن
        self.optimizer.update(self) 
        
        return loss
    
    def fit(self, X_train, Y_train, epochs, batch_size=1, verbose=100):
        N = X_train.shape[0]
        history = []
        
        for e in range(epochs):
            loss = 0
            for i in range(0, N, batch_size):
                x_batch = X_train[i:i+batch_size]
                y_batch = Y_train[i:i+batch_size]
                
                loss += self.train_step(x_batch, y_batch)
            
            avg_loss = loss / N
            history.append(avg_loss)
            
            if verbose and (e % verbose == 0 or e == epochs - 1):
                print(f'Epoch {e+1}/{epochs} - Loss: {avg_loss:.6f}')
        
        return history

    def predict(self, x):
        return self.forward(x)