import numpy as np
from models import LinearLayer

class LinearLayerGaussianProposalGenerator:
    """Generates proposal parameters for a linear layer (with parameters w and b).
    
    This object creates proposals by drawing from Gaussians centered at each parameter with some
    user-specified `scale` that controls the variance in the distribution.

    The `decay` parameter controls how the standard deviation of the proposal distribution shrinks
    with time. With each successive call of `get_param_proposal`, self.scale = self.scale * decay.
    Defaults to 1 (no decaying).
    """
    def __init__(self, scale: float = 1., decay: float = 1.):
        self.scale = scale
        self.decay = decay

    def get_param_proposal(self, model: LinearLayer) -> dict:
        """Get a parameter proposal from current model state."""
        self.scale *= self.decay

        proposed_w = np.random.normal(loc=model.w, scale=self.scale)
        proposed_b = np.random.normal(loc=model.b, scale=self.scale)

        return {'w': proposed_w, 'b': proposed_b}
    
class TwoLayerNNGaussianProposalGenerator:
    """Generates proposal parameters for a 2-layer NN (with parameters w1, b1, w2, and b2).
    
    This object creates proposals by drawing from multivariate Gaussians centered at the current value
    of each parameter with some user-specified `scale` that controls the covariance in the distribution.

    The `decay` parameter controls how the covariance of the proposal distribution shrinks
    with time. With each successive call of `get_param_proposal`, self.scale = self.scale * decay.
    Defaults to 1 (no decaying).
    """
    def __init__(self, scale: float = 1., decay: float = 1.):
        self.scale = scale
        self.decay = decay

    def get_param_proposal(self, model) -> dict:
        """Get a parameter proposal from current model state."""
        self.decay *= self.decay

        proposed_w1 = np.random.normal(model.w1, self.scale)
        proposed_w2 = np.random.normal(model.w2, self.scale)

        proposed_b1 = np.random.normal(model.b1, self.scale)
        proposed_b2 = np.random.normal(model.b2, self.scale)

        return {
            'w1': proposed_w1,
            'b1': proposed_b1,
            'w2': proposed_w2,
            'b2': proposed_b2,
        }