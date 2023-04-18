import numpy as np
from sklearn.metrics import log_loss

def softmax(xs):
    return np.exp(xs)/(np.sum(np.exp(xs)))
    
def loss(nn,w1,b1,w2,b2,xs,ys):
    pred_probs = []
    N = len(ys)
    for i,x in enumerate(xs):
        probs = softmax(nn.forward(x,w1,b1,w2,b2))
        pred_probs.append(probs)
    return log_loss(ys,pred_probs,labels=[0,1,2,3,4,5,6,7,8,9])