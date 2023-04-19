import numpy as np
from sklearn.metrics import log_loss

def softmax(xs,temperature=1e3):
    if len(xs.shape) == 1: #Increase dimension to fit below for 1 dimension arrays
        xs = np.expand_dims(xs,0)
    norm_fact = np.max(xs)
    temperature = norm_fact   #I'm not sure if this is the best way? Should I do it individually for each set of probabilities?
    return np.exp(xs/temperature)/(np.sum(np.exp(xs/temperature),axis=1).reshape(-1,1))
    
def cross_entropy_loss(targets,preds):
    return log_loss(targets,preds,labels=[0,1,2,3,4,5,6,7,8,9])