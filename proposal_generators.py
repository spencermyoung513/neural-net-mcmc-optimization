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
    
class TwoLayerNNGibbsProposalGenerator:
    """Generates proposal parameters for a 2-layer NN (with parameters w1, b1, w2, and b2).
    
    This object creates proposals by randomly selecting `pct_entries_to_change` of the elements of
    each weight/bias and adjusting them by drawing from Gaussians centered at their current value
    with some user-specified `scale` that controls the covariance in the distribution.

    The `decay` parameter controls how the covariance of the proposal distribution shrinks
    with time. With each successive call of `get_param_proposal`, self.scale = self.scale * decay.
    Defaults to 1 (no decaying).
    """
    def __init__(self, pct_entries_to_change: float = 0.1, scale: float = 1., decay: float = 1.):
        self.pct_entries_to_change = pct_entries_to_change
        self.scale = scale
        self.decay = decay

    def get_param_proposal(self, model) -> dict:
        """Get a parameter proposal from current model state."""
        self.decay *= self.decay

        probs = ((1 - self.pct_entries_to_change), self.pct_entries_to_change)

        w1_mask = np.random.choice([True, False], size=model.w1.shape, p=probs)
        b1_mask = np.random.choice([True, False], size=model.b1.shape, p=probs)
        w2_mask = np.random.choice([True, False], size=model.w2.shape, p=probs)
        b2_mask = np.random.choice([True, False], size=model.b2.shape, p=probs)

        new_w1 = np.random.normal(model.w1, self.scale)
        proposed_w1 = model.w1.copy()
        proposed_w1[w1_mask] = new_w1[w1_mask]

        new_b1 = np.random.normal(model.b1, self.scale)
        proposed_b1 = model.b1.copy()
        proposed_b1[b1_mask] = new_b1[b1_mask]

        new_w2 = np.random.normal(model.w2, self.scale)
        proposed_w2 = model.w2.copy()
        proposed_w2[w2_mask] = new_w2[w2_mask]

        new_b2 = np.random.normal(model.b2, self.scale)
        proposed_b2 = model.b2.copy()
        proposed_b2[b2_mask] = new_b2[b2_mask]

        return {
            'w1': proposed_w1,
            'b1': proposed_b1,
            'w2': proposed_w2,
            'b2': proposed_b2,
        }