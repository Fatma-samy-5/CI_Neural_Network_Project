import numpy as np

class Activation:
    def forward(self, x):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

class Tanh(Activation):
    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, output_gradient):
        return output_gradient * (1 - np.power(np.tanh(self.x), 2))

class Sigmoid(Activation):
    def forward(self, x):
        self.x = x
        self.output = np.tanh(x)
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (1 - np.power(self.output, 2))