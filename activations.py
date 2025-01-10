import numpy as np

class ReLU:
    def forward(self, input):
        self.output = np.maximum(input, 0) # performing element wise over a vector
        return self.output

    def backward(self, output_gradient):
        return output_gradient * (self.output > 0) # 

class Sigmoid:
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output
    
    def backward(self, output_gradient):
        return output_gradient * self.output * (1 - self.output)