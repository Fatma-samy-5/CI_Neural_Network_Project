import numpy as np

class Loss:
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def gradient(self, y_true, y_pred):
        raise NotImplementedError

class MeanSquaredError(Loss):
   
    def loss(self, y_true, y_pred):
        return 0.5 * np.mean(np.power(y_true - y_pred, 2)) 

    #(dL/dY_pred = 1/N * (Y_pred - Y_true))
    def gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.size