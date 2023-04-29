"""
Loss function factory class for the XGBoost model. Originally developed in ml.v7.ipynb.

April 27, 2023
"""

from joblib import Parallel, delayed
import numpy as np
from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean

    
class LossFactory:

    def __init__(self, split_size, loss_type = 'mse', gamma = 1.0, mse_weight = None, dtw_weight = None):
        
        if loss_type not in ['mse', 'dtw', 'combined']:
            raise ValueError(f'Loss type [{loss_type}] not supported')
        
        self.split_size = split_size  # number of samples considered in computing batch loss
        self.gamma = gamma  # hyperparam for SDTW loss, didn't really tune this so there's potential room for improvement
        self.mse_weight = mse_weight  # weight for MSE loss if combined loss is used
        self.dtw_weight = dtw_weight  # weight for SDTW loss if combined loss is used

        if loss_type == 'mse':
            self.loss_function = self.mse_loss
        elif loss_type == 'dtw':
            self.loss_function = self.soft_dtw_loss
        elif loss_type == 'combined':
            self.loss_function = self.combined_loss
        
    def __call__(self, y_pred, data):
        return self.loss_function(y_pred, data)

    def get_function(self):
        """
        Returns the loss function created by the factory.
        """
        return self.loss_function

    def mse_loss(self, y_pred, data):
        """
        Normal MSE loss.
        """
        
        # get the labels from the data
        y_true = data.get_label()

        # get the number of batches that will be split from the number of samples
        # in the given data and prediction vector
        num_batches = int(len(y_pred) / self.split_size)
        
        # array for collecting the error before calculating the gradients and hessians
        errs = np.zeros_like(y_true)

        # iterate through the batches and calculate the error for each batch
        # NOTE: it isn't actually necessary to do this for MSE, but it stays consistent with
        # how the SDTW loss and any other of the time series loss functions are implemented
        for i in range(num_batches):
            y_true_curr = y_true[i * self.split_size: (i + 1) * self.split_size]
            y_pred_curr = y_pred[i * self.split_size: (i + 1) * self.split_size]
            err = y_true_curr - y_pred_curr
            errs[i * self.split_size: (i + 1) * self.split_size] = err

        grad = -2 * errs
        hess = 2 * np.ones_like(y_true)
        return grad, hess

    def soft_dtw_loss(self, y_pred, data):
        """
        Soft DTW loss calculated in batches.
        """

        def batch_loss_helper(i, y_true, y_pred, split_size):
            """
            Helper method calculating the SDTW loss for a single batch.
            """
            
            y_true_curr = y_true[i * split_size: (i + 1) * split_size]
            y_pred_curr = y_pred[i * split_size: (i + 1) * split_size]

            grad_curr, hess_curr = self.compute_sdtw(y_true_curr, y_pred_curr)
            grad_curr = grad_curr.flatten()
            hess_curr = hess_curr.flatten()

            return grad_curr, hess_curr

        # get the labels from the data and comput the number of batches
        y_true = data.get_label()
        num_batches = int(len(y_pred) / self.split_size)

        # init arrays to store gradients and bessians for SDTW
        grads = np.zeros_like(y_true)
        hesses = np.zeros_like(y_true)
        
        # parallelize SDTW loss batchwise bc it's slow
        results = Parallel(n_jobs = -1)(
            delayed(batch_loss_helper)(
                i, y_true, y_pred, self.split_size
            ) for i in range(num_batches)
        )

        # populate the grads and hesses arrays
        for i, (grad_curr, hess_curr) in enumerate(results):
            grads[i * self.split_size: (i + 1) * self.split_size] = grad_curr
            hesses[i * self.split_size: (i + 1) * self.split_size] = hess_curr

        return grads, hesses
    
    def compute_sdtw(self, y_true, y_pred):
        """
        Computes the SDTW loss and returns the gradients and hessians.

        NOTE: Hessian computation was so expensive it made the training process
        far too slow. However, I did not try to use machine(s) more powerful than
        my own laptop. There's room for improvement here by using Autograd to compute
        the actual hessians and getting more computational horsepower involved for
        training.
        """

        x = y_true.reshape(-1, 1)
        y = y_pred.reshape(-1, 1)
        D = SquaredEuclidean(x, y)
        sdtw = SoftDTW(D, gamma = self.gamma)
        sdtw.compute()
        E = sdtw.grad()
        G = D.jacobian_product(E)

        # returned hessian
        return G, np.ones(len(G))
    
    def combined_loss(self, y_pred, data):
        """
        For combining SDTW and MSE loss.
        """

        if self.mse_weight is None or self.dtw_weight is None:
            raise ValueError('mse_weight and dtw_weight must be set before calling combined_loss')

        mse_grads, mse_hesses = self.mse_loss(y_pred, data)
        dtw_grads, dtw_hesses = self.soft_dtw_loss(y_pred, data)

        combined_grad = self.mse_weight * mse_grads + self.dtw_weight * dtw_grads
        combined_hess = self.mse_weight * mse_hesses + self.dtw_weight * dtw_hesses

        return combined_grad, combined_hess