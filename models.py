import numpy as np

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
