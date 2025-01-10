import numpy as np

class Linear:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias # dot product multiplication + bias to determine output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, np.transpose(self.weights))
        weight_gradient = np.dot(np.transpose(self.input), output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        
        self.weights = self.weights - learning_rate*weight_gradient 
        self.bias = self.bias - learning_rate*bias_gradient

        return input_gradient 
    
    
class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None # matrix to represent dropped out inputs of the layer

    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(1, 1-self.dropout_rate, size=input.shape) / (1 - self.dropout_rate)
            return input*self.mask # dropped out neurons squashed to zero
        else:
            return input # during testing, all neurons persist
    
    def backward(self, output_gradient):
        return output_gradient*self.mask