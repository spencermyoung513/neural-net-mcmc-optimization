import numpy as np

class BatchSampler:

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def get_random_sample(self, size: int = 25):
        sample = np.random.choice(len(self.X), size)
        return self.X[sample], self.y[sample]