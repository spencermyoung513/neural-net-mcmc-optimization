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
    
class two_model:
    "A two layer neural network with Relu Activation"
    def __init__(self,w1,b1,w2,b2):
		if w1 is None:
        	self.w1 = np.random.normal(loc=0,scale=1.0,size=(30,64))
        	self.b1 = np.random.normal(loc=0,scale=1.0,size=50)
        	self.w2 = np.random.normal(loc=0,scale=1.0,size=(10,30))
        	self.b2 = np.random.normal(loc=0,scale=1.0,size=10)
		else:
			self.w1 = w1
        	self.b1 = b1
        	self.w2 = w2
        	self.b2 = b2

    def forward(self,x,w1,b1,w2,b2):
        x = x.flatten()
        return w2@np.maximum(w1@x + b1,0)+b2

    def update_weights(self,w1,b1,w2,b2):
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
