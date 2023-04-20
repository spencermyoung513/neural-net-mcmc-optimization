import numpy as np
from typing import Tuple, Dict

def optimize_via_mcmc(model, initial_params, data_sampler, proposal_generator, loss_function,
                      beta: float = 0.5, num_iterations: int = 5000, batch_size: int = 25,regularize: bool = False, alpha: float = 2) -> Tuple[Dict, Dict]:
    """Use MCMC sampling to sample from likely optimizers of a loss surface.
    
    Args:
        model (Object): The model whose parameters are being optimized (should be an uninstantiated Python class with a `forward` method).
        initial_params (dict): Dictionary specifying initial values of model parameters
        data_sampler (Object): Object with a `get_random_sample` method that returns a batch of data
        proposal_generator (Object): Object with a `get_param_proposal` method that generates parameter proposals
        loss_function (callable): Method that computes a loss score based on model predictions
        beta (float): Hyperparameter used to define MCMC acceptance probability
        num_iterations (int): Number of iterations to run MCMC sampling for
        batch_size (int): Number of data points to return from data_sampler

    Returns:
        best_params (dict): Best computed set of parameters
        history (dict): Dictionary with "training" history (keys are `acceptance_ratio`, `parameter_values`, and `loss_values`)
    """
    current_model = model(**initial_params)
    parameter_values = {param: [value] for param, value in initial_params.items()}
    
    num_accepted = 0
    min_achieved_loss = np.inf
    best_params = None
    loss_values = []

    for i in range(num_iterations):

        # Get data batch and proposed model.
        X_sample, y_sample = data_sampler.get_random_sample(batch_size)
        proposed_params = proposal_generator.get_param_proposal(current_model)
        proposed_model = model(**proposed_params)

        # Get predictions of proposed model and current model on batch.
        proposed_model_preds = proposed_model.forward(X_sample)
        current_model_preds = current_model.forward(X_sample)

        # Compute loss on batch of proposed and current model.
        if regularize:
            proposal_loss = loss_function(y_sample,proposed_model_preds)\
            +alpha*(np.linalg.norm(proposed_model.w1)+np.linalg.norm(proposed_model.w2))
            current_loss = loss_function(y_sample,current_model_preds)\
            +alpha*(np.linalg.norm(current_model.w1)+np.linalg.norm(current_model.w2))
        else:
            proposal_loss = loss_function(y_sample, proposed_model_preds)
            current_loss = loss_function(y_sample, current_model_preds)
            

        # Decide whether/not to accept the proposal.

        acceptance_probability = np.exp(-beta * (proposal_loss - current_loss))
        accept = np.random.rand() <= acceptance_probability

        if accept:

            loss_values.append(proposal_loss)

            if proposal_loss < min_achieved_loss:
                min_achieved_loss = proposal_loss
                best_params = proposed_params

            num_accepted += 1
            current_model = proposed_model

            for param, val in proposed_params.items():
                parameter_values[param].append(val)

    history = {
        'acceptance_ratio': num_accepted / num_iterations,
        'parameter_values': parameter_values,
        'loss_values': loss_values,
    }

    return best_params, history