import numpy as np
from functions import softmax,softmax2,softmax3
class LinearLayer:
    """A one-layer linear network with no activation."""
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        return np.dot(x, self.w) + self.b
    
    def update_weights(self, w, b):
        self.w = w
        self.b = b
        return self
    
class TwoLayerNN:
    """A two layer neural network with Relu Activation"""
    def __init__(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2

    def forward(self,x):
        return softmax3((self.w2 @ np.maximum((self.w1 @ x.T).T + self.b1, 0).T).T + self.b2)

    def update_weights(self, w1, b1, w2, b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
