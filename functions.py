import numpy as np
from sklearn.metrics import log_loss

def softmax(xs):
    if len(xs.shape) == 1: #Increase dimension to fit below for 1 dimension arrays
        xs = np.expand_dims(xs,0)
    norm_fact = np.max(xs)
    temperature = norm_fact
    return np.exp(xs/temperature)/(np.sum(np.exp(xs/temperature),axis=1).reshape(-1,1))

def softmax2(xs):
    if len(xs.shape) == 1: #Increase dimension to fit below for 1 dimension arrays
        xs = np.expand_dims(xs,0)
    norm_fact = np.amax(np.abs(xs),axis=1)
    temperature = norm_fact.reshape(-1,1)
    return np.exp(xs/temperature)/(np.sum(np.exp(xs/temperature),axis=1).reshape(-1,1))

def softmax3(xs):
    if len(xs.shape) == 1: #Increase dimension to fit below for 1 dimension arrays
        xs = np.expand_dims(xs,0)
    norm_fact = np.amax(np.abs(xs),axis=1)
    mean = np.mean(xs,axis=1).reshape(-1,1)
    temperature = norm_fact.reshape(-1,1)
    return np.exp((xs-mean)/temperature)/(np.sum(np.exp((xs-mean)/temperature),axis=1).reshape(-1,1))
    
def cross_entropy_loss(targets,preds):
    return log_loss(targets,preds,labels=[0,1,2,3,4,5,6,7,8,9])