import numpy as np
from models import LinearLayer
from scipy.stats import multivariate_normal as mv_norm

class LinearLayerUnivariateNormalProposalGenerator:
    """Generates proposal parameters for a univariate linear layer (with parameters w and b).
    
    This object creates proposals by drawing from Gaussians centered at each parameter with some
    user-specified `stdev` that controls the variance in the distribution.

    The `decay` parameter controls how the standard deviation of the proposal distribution shrinks
    with time. With each successive call of `get_param_proposal`, self.stdev = self.stdev * decay.
    Defaults to 1 (no decaying).
    """

    def __init__(self, stdev: float = 1., decay: float = 1.):
        self.stdev = stdev
        self.decay = decay

    def get_param_proposal(self, model: LinearLayer) -> dict:
        """Get a parameter proposal from current model state."""
        self.stdev *= self.decay

        proposed_w = np.random.normal(loc=model.w, scale=self.stdev)
        proposed_b = np.random.normal(loc=model.b, scale=self.stdev)

        return {'w': proposed_w, 'b': proposed_b}
    
class LinearLayerMultivariateNormalProposalGenerator:
    """Generates proposal parameters for a univariate linear layer (with parameters w and b).
    
    This object creates proposals by drawing from multivariate Gaussians centered at each parameter
    with some user-specified `cov` that controls the covariance in the distribution.

    The `decay` parameter controls how the covariance of the proposal distribution shrinks
    with time. With each successive call of `get_param_proposal`, self.cov = self.cov * decay.
    Defaults to 1 (no decaying).
    """
    def __init__(self, cov: float = 1., decay: float = 1.):
        self.cov = cov
        self.decay = decay

    def get_param_proposal(self, model: LinearLayer) -> dict:
        """Get a parameter proposal from current model state."""
        self.cov *= self.decay

        proposed_w = mv_norm.rvs(model.w, self.cov)
        proposed_b = mv_norm.rvs(model.b, self.cov)

        return {'w': proposed_w, 'b': proposed_b}