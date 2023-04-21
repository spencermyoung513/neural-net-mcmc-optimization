import pickle
import numpy as np
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from mcmc_optimization import optimize_via_mcmc
from models import TwoLayerNN, LinearLayer
from data_samplers import BatchSampler
from proposal_generators import (
    LinearLayerGaussianProposalGenerator,
    TwoLayerNNGaussianProposalGenerator,
    TwoLayerNNGibbsProposalGenerator
)
from functions import cross_entropy_loss


def run_mcmc_optimization_on_mnist(model, initial_model_params: dict, mcmc_params: dict, proposal_generator,
                                   num_trials: int = 10, save_path: str = None) -> dict:
    """Run MCMC optimization on the MNIST dataset `num_trials` number of times and record results.

    Args:
        model: The model whose parameters are being optimized (uninstantiated).
        initial_model_params: Initial parameters for the model to optimize.
        mcmc_params: Initial parameters for the MCMC optimization process.
        proposal_generator: Parameter proposal generator object (uninstantiated).
        num_trials (int): Total number of times to run MCMC optimization.
        save_path (str): If provided, tells method to save results to `save_path`.

    Returns:
        results (dict): Results of MCMC runs
    """
    digits_X, digits_y = load_digits(return_X_y=True)

    test_accuracies = []
    training_times = []
    histories = []
    best_found_params = None
    best_test_accuracy = 0

    for _ in range(num_trials):
        X_train, X_test, y_train, y_test = train_test_split(digits_X, digits_y, test_size=0.2)
        data_sampler = BatchSampler(X=X_train, y=y_train)

        start = time()
        best_params, history = optimize_via_mcmc(
            model,
            initial_model_params,
            data_sampler,
            proposal_generator,
            cross_entropy_loss,
            **mcmc_params
        )
        time_elapsed = time() - start

        best_model = model(**best_params)

        num_correct = 0
        for image, actual_label in zip(X_test, y_test):
            prediction = np.argmax(best_model.forward(image))
            if prediction == actual_label:
                num_correct += 1
        test_accuracy = num_correct / len(y_test)

        test_accuracies.append(test_accuracy)
        training_times.append(time_elapsed)
        histories.append(history)

        if test_accuracy > best_test_accuracy:
            best_found_params = best_params
            best_test_accuracy = test_accuracy

    results = {
        'test_accuracies': test_accuracies,
        'training_times': training_times,
        'histories': histories,
        'best_found_params': best_found_params,
    }

    if save_path is not None:
        with open(save_path, mode='xb') as file:
            pickle.dump(results, file)

    return results


if __name__ == "__main__":

    hidden_size = 30
    final_size = 10

    w1 = np.random.normal(loc=0, scale=1.0, size=(hidden_size, 64))
    b1 = np.random.normal(loc=0, scale=1.0, size=hidden_size)
    w2 = np.random.normal(loc=0, scale=1.0, size=(final_size, hidden_size))
    b2 = np.random.normal(loc=0, scale=1.0, size=final_size)

    initial_model_params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}
    mcmc_params = {'beta': 100, 'num_iterations': 20000, 'batch_size': 50}

    gibbs_proposal_generator = TwoLayerNNGibbsProposalGenerator(pct_entries_to_change=0.1, scale=5, decay=1-1e-7)
    gaussian_proposal_generator = TwoLayerNNGaussianProposalGenerator(scale=5, decay=1-1e-7)

    run_mcmc_optimization_on_mnist(TwoLayerNN, initial_model_params, mcmc_params, gibbs_proposal_generator, num_trials=10, save_path='nn_gibbs_results.pkl')
    run_mcmc_optimization_on_mnist(TwoLayerNN, initial_model_params, mcmc_params, gaussian_proposal_generator, num_trials=10, save_path='nn_gaussian_results.pkl')