import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Tanh/Sigmoid
        limit = np.sqrt(2 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        

        self.bias = np.zeros((1, output_size))

        self.weights_gradient = None
        self.bias_gradient = None

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, output_gradient):
       
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        input_gradient = np.dot(output_gradient, self.weights.T)
        
        return input_gradient