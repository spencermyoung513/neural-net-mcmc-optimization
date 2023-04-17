import numpy as np
from models import LinearLayer

class LinearLayerUnivariateNormalProposalGenerator:
    """Generates proposal parameters for a univariate linear layer (with parameters w and b).
    
    This object creates proposals by drawing from Gaussians centered at each parameter with some
    user-specified `stdev` that controls the variance in the distribution.
    """

    def __init__(self, stdev: float = 1.1):
        self.stdev = stdev

    def get_param_proposal(self, model: LinearLayer) -> dict:
        """Get a parameter proposal from current model state."""
        proposed_w = np.random.normal(loc=model.w, scale=self.stdev)
        proposed_b = np.random.normal(loc=model.b, scale=self.stdev)

        return {'w': proposed_w, 'b': proposed_b}