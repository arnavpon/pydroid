"""
Wrapper class for the XGBoost wrapper, called MoodBoost since it's for Mood Triggers.

April 27, 2023
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy.signal import resample
import xgboost as xgb

from pipeline_v1.losses import LossFactory
from pipeline_v1.peaks import get_peaks_for_hr
from pipeline_v1.signal_pross import bandpass, get_hrv, min_max_scale, n_moving_avg


class MoodBoost:
    
    def __init__(self, truths, label_col = 'bvp', subject_col = 'subject',
                
                # for model customization
                random_state = None, loss_type = 'mse', excluded_subject = None,
                
                # hyperparameters
                n_estimators = 100, split_size = 1280, learning_rate = 0.1, test_size = 0.3, early_stopping_rounds = 50,
                mse_weight = None, dtw_weight = None, data_beg = 1000, data_end = 10000, batches = 1, finetune = True,
                min_bandpass_freq = 0.7, max_bandpass_freq = 4.0, bandpass_order = 4,
                predicted_peaks_prominence = 0.22, true_peaks_prominence = 0.15,
                
                # hyperparams from XGBoost docs
                max_depth = 7, max_bin = 255,
                num_feats_per_channel = 3, skip_amount = 15):
        """
        Truths parameter is an array of IeeeGroundTruth objects from the truth.py module.
        This class wraps the XGBoost model and various supporting functionality.
        """
        
        self.label_col = label_col  # column name for the ground truth labels
        self.subject_col = subject_col  # column name for subject IDs

        self.random_state = random_state
        self.excluded_subject = excluded_subject  # subject to exclude from the training data

        self.n_estimators = n_estimators
        self.split_size = split_size  # number of samples considered in computing batch loss
        self.learning_rate = learning_rate
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds
        self.finetune = finetune

        # sample idxs to begin from for use in the training set; this is to limit data per subject and avoid 
        # noise from beginning and end of video data
        self.data_beg = data_beg 
        self.data_end = data_end
        
        # bandpass/HR calculation hyperparams
        self.min_bandpass_freq = min_bandpass_freq
        self.max_bandpass_freq = max_bandpass_freq
        self.bandpass_order = bandpass_order
        self.predicted_peaks_prominence = predicted_peaks_prominence
        self.true_peaks_prominence = true_peaks_prominence

        # XGBoost hyperparams
        self.max_depth = max_depth
        self.max_bin = max_bin

        # hyperparams for memory features from raw RGB
        self.num_feats_per_channel = num_feats_per_channel
        self.skip_amount = skip_amount

        self.model = None
        self.training_loss = None
        self.test_loss = None

        # process IEEE data; exclude subject if specified
        self.given_data = self.prepare_dataset_from_subjects(truths, data_beg = data_beg, data_end = data_end)
        print(f'Number of samples in dataset: {len(self.given_data)}')
        if self.excluded_subject is not None:
            self.given_data = self.given_data[self.given_data[self.subject_col] != self.excluded_subject]
        
        # get feature names as an array
        self.features = np.array(self.given_data.drop(columns = [self.label_col, self.subject_col]).columns)
        
        # set random state, if specified
        if self.random_state is not None:
            random.seed(self.random_state)
        
        # split data into training and test batches
        splits = self.split_data()

        # randomly sample splits for the training set; split into the training and test sets
        self.train_split_indices = random.sample(range(len(splits)), int(len(splits) * (1 - self.test_size)))
        self.train_splits = [splits[i] for i in self.train_split_indices]
        self.test_splits = [splits[i] for i in range(len(splits)) if i not in self.train_split_indices]
        
        # list of training data as batches
        # NOTE: If we just set the batches parameter to 1, then the whole training process
        # just effectively proceeds unbatched
        self.train_data = []

        # get batch size from split size and given number of batches
        batch_size = len(self.train_splits) // batches
        
        # create batches by applying a similar splitting method to just the 
        # training data as is used in the self.split_data() method
        for _ in range(batches):
            
            # curr batch is a random sample of the training splits
            batch_split_idxs = random.sample(range(len(self.train_splits)), batch_size)
            
            # ramdomly selected splits go in the batch and the rest stay in the training set
            batch_splits = [self.train_splits[i] for i in batch_split_idxs]
            self.train_splits = [self.train_splits[i] for i in range(len(self.train_splits)) if i not in batch_split_idxs]

            # create batch data and label
            batch_indices = [idx for split in batch_splits for idx in split]
            batch_rows = self.given_data.iloc[batch_indices].drop(columns = [self.subject_col])
            batch_X = batch_rows.drop(columns = [self.label_col]).to_numpy()
            batch_y = batch_rows[self.label_col].to_numpy()

            # save batch X and y as tuples for making a DMatrix later
            self.train_data.append((batch_X, batch_y))

        # create the testing dataset
        test_indices = [idx for split in self.test_splits for idx in split]
        test_rows = self.given_data.iloc[test_indices].drop(columns = [self.subject_col])
        self.test_X = test_rows.drop(columns = [self.label_col]).to_numpy()
        self.test_y = test_rows[self.label_col].to_numpy()

        # save the testing dataset as a DMatrix
        self.test_data = xgb.DMatrix(self.test_X, self.test_y)

        # initialize the loss function for the model
        self.loss = LossFactory(self.split_size, loss_type = loss_type, mse_weight = mse_weight, dtw_weight = dtw_weight).get_function()
        if self.finetune:
            self.mse_loss = LossFactory(self.split_size, loss_type = 'mse').get_function()
    
    def split_data(self, to_exclude = None):
        """
        Split data into training and testing while preserving consecutive samples from the same subject.
        """

        # get indices of samples from each subject
        subject_indices = self.given_data.groupby(self.subject_col).indices
        
        splits = []
        for _, indices in subject_indices.items():
            
            n_splits = len(indices) // self.split_size
            if n_splits > 0:

                subject_splits = []
                for i in range(n_splits):
                    split_start = i * self.split_size
                    split_end = (i + 1) * self.split_size
                    subject_split = indices[split_start: split_end]
                    subject_splits.append(subject_split)
                
                splits.extend(subject_splits)
        
        return splits
    
    def fit(self):
        """
        Train the model.
        """

        # for tracking how long the training process takes
        t1 = datetime.today()
        
        # init params for the XGBoost model
        self.params = {
            'learning_rate': self.learning_rate,
            'booster': 'gbtree',
            'max_depth': self.max_depth,
            'max_bin': self.max_bin,
        }

        met = {}

        for i, (batch_X, batch_y) in enumerate(self.train_data):
            print(f'\n\nOn batch {i + 1} of {len(self.train_data)}:')

            batch_dmatrix = xgb.DMatrix(batch_X, batch_y)
            
            # train the model
            self.model = xgb.train(
                self.params,
                batch_dmatrix,
                num_boost_round = self.n_estimators,
                early_stopping_rounds = self.early_stopping_rounds,
                feval = self.hr_error_eval_metric,
                verbose_eval = 5,
                evals = [(batch_dmatrix, 'train'), (self.test_data, 'test')],
                xgb_model = self.model,  # for providing initial parameters from which to start (or None, if it's first iteration)
                obj = self.loss,
                evals_result = met
            )

            if self.finetune:
                print('Fintuning...')
                
                # make a copy of the model and get predictions on the current batch
                model_copy = self.model.copy()
                pred = model_copy.predict(batch_dmatrix)

                # initialize an array to hold the new, adjqusted targets for finetuning step
                new_targ = np.ones(len(pred))
                
                # get the number of splits in the current batch
                nsplits = len(pred) // self.split_size
                
                # use the difference between the labels and predictions as the new targets
                for i in range(nsplits):
                    pred_curr = pred[i * self.split_size: (i + 1) * self.split_size]
                    label_curr = batch_y[i * self.split_size: (i + 1) * self.split_size]
                    new_targ[i * self.split_size: (i + 1) * self.split_size] = label_curr - pred_curr

                # create a new DMatrix with the adjusted targets
                new_batch_data = xgb.DMatrix(batch_X, new_targ)

                self.model = xgb.train(
                    self.params,
                    new_batch_data,
                    num_boost_round = self.n_estimators // 2,
                    early_stopping_rounds = self.early_stopping_rounds // 2,
                    feval = self.hr_error_eval_metric,
                    verbose_eval = 5,
                    evals = [(new_batch_data, 'train'), (self.test_data, 'test')],
                    xgb_model = model_copy,
                    obj = self.mse_loss
                )

        self.training_loss = met['train']['hr_err']
        self.test_loss = met['test']['hr_err']
        print(f'Finished training in {datetime.today() - t1}')

    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, model_file = 'moodboost.pkl'):
        self.model.save_model(model_file)

    def eval(self):
        """
        Compute validation error for the model.
        """
        
        # get number of splits in the test set
        nsplits = int(len(self.test_X) / self.split_size)
        
        # init arrays for storing hr errors and mses
        errs = []
        mses = np.zeros(len(self.test_X))
        
        for i in range(nsplits):
            
            # get the predictions and ground truth for this batch
            curr_pred = self.predict(xgb.DMatrix(self.test_X[i * self.split_size: (i + 1) * self.split_size, :]))
            curr_true = self.test_y[i * self.split_size: (i + 1) * self.split_size]
            
            # process both the predicted and ground truth signal
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 5, use_bandpass = True)
            
            # get the errors
            mses[i * self.split_size: (i + 1) * self.split_size] = curr_true - curr_pred
            hr_err = self.get_hr_error(curr_true, curr_pred, square = False)
            errs.append(hr_err)
        
        return np.mean(np.square(mses)), np.mean(errs), np.mean(np.square(errs))

    def validate(self, val_data = None):
        """
        Get validation error for the model for use in cross-validation.
        """

        if val_data is None:
            valX = self.test_X
            valy = self.test_y
        else:
            valX, valy = val_data

        # get number of splits in the test set
        nsplits = int(len(valX) / self.split_size)
        
        # init array for collecting errors
        errors = []
        for i in range(nsplits):
            
            curr_X = xgb.DMatrix(valX[i * self.split_size: (i + 1) * self.split_size, :])
            curr_pred = self.predict(curr_X)
            curr_true = valy[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 5, use_bandpass = True)
            
            # get each error and aggregate into a dict
            mse = np.mean(np.square(curr_true - curr_pred))
            hr_err = self.get_hr_error(curr_true, curr_pred, square = False)
            hrv_err = self.get_hrv_error(curr_true, curr_pred, square = False)
            peaks_err = self.get_peaks_error(curr_true, curr_pred, square = False)
            errors.append({
                'mse': mse,
                'hr_err': hr_err,
                'hrv_err': hrv_err,
                'peaks_err': peaks_err
            })

        return errors

    def plot_hr_loss(self):
        """
        Plot the training and test HR loss for the model across each boosting round. 
        """
            
        if self.training_loss is not None and self.test_loss is not None:
            plt.plot(self.training_loss, label = 'training loss')
            plt.plot(self.test_loss, label = 'test loss')
            plt.legend()
        
    def get_model_stats(self):
        """
        Get the model stats, including the best test loss, the min and max tree depths, and feature importances.
        """

        print(f'Best test loss: {min(self.test_loss)}\n')
        print('\nFeature importances:')
        print(self.get_feature_importances())
    
    def get_feature_importances(self):
        feature_importances = self.model.get_score(importance_type = 'gain')
        feature_importances = sorted(feature_importances.items(), key = lambda kv: kv[1], reverse = True)
        return pd.DataFrame(feature_importances, columns = ['feature', 'importance'])
    
    def hr_error_eval_metric(self, y_pred, eval_data):
        """
        Format HR error as an eval metric.
        """
        
        y_true = eval_data.get_label()
        nsplits = int(len(y_pred) / self.split_size)
        hr_err = []
        
        for i in range(nsplits):
            curr_pred = y_pred[i * self.split_size: (i + 1) * self.split_size]
            curr_true = y_true[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 10, use_bandpass = True)
            hr_err.append(self.get_hr_error(curr_true, curr_pred, square = False))
        
        return 'hr_err', np.mean(hr_err)
    
    def get_hr_error(self, y_true, y_pred, square = True):
        """
        Get the raw HR error. Optional squaring.
        """

        true_peaks, _ = self.get_true_peaks(y_true)
        pred_peaks, _ = self.get_predicted_peaks(y_pred)

        if len(true_peaks) >= 2:
            true_ibis = np.diff(true_peaks) / 64
            true_hr = 60 / np.mean(true_ibis)
        else:
            true_hr = 0

        if len(pred_peaks) >= 2:
            pred_ibis = np.diff(pred_peaks) / 64
            pred_hr = 60 / np.mean(pred_ibis)
        else:
            pred_hr = 0
        
        if square:
            return np.power(true_hr - pred_hr, 2)
        return abs(true_hr - pred_hr)
    
    def get_peaks_error(self, y_true, y_pred, square = True):
        """
        Get the "peaks error." This is just the difference in the number of peaks
        detected between the predictions and the ground truth.
        """

        true_peaks, _ = self.get_true_peaks(y_true)
        pred_peaks, _ = self.get_predicted_peaks(y_pred)
        if square:
            return np.power(len(true_peaks) - len(pred_peaks), 2)
        return abs(len(true_peaks) - len(pred_peaks))
    
    def get_hrv_error(self, y_true, y_pred, square = True):
        """
        Error metric for HRV.
        """

        true_peaks, _ = self.get_true_peaks(y_true)
        pred_peaks, _ = self.get_predicted_peaks(y_pred)

        if len(true_peaks) >= 2:
            true_ibis = np.diff(true_peaks) / 64
            true_hrv = get_hrv(true_ibis)
        else:
            true_hrv = 0

        if len(pred_peaks) >= 2:
            pred_ibis = np.diff(pred_peaks) / 64
            pred_hrv = get_hrv(pred_ibis)
        else:
            pred_hrv = 0
        
        if square:
            return np.power(true_hrv - pred_hrv, 2)
        return abs(true_hrv - pred_hrv)
    
    def process_signal(self, y_true, y_pred, smoothing_window = 10, use_bandpass = False):
        """
        Process predictions and ground truth signal together. Note that the same processing isn't
        applied to both signals.
        """

        # process the predictions
        if y_pred is not None:
            orig_len = len(y_pred)
            y_pred = n_moving_avg(y_pred, smoothing_window)
            y_pred = resample(y_pred, orig_len)
            if use_bandpass:
                y_pred = bandpass(y_pred, 64, [self.min_bandpass_freq, self.max_bandpass_freq], self.bandpass_order)
            y_pred = min_max_scale(y_pred)
        
        # process the ground truth
        if y_true is not None:
            y_true = n_moving_avg(y_true, 20)
            y_true = resample(y_true, orig_len)
            if use_bandpass:
                y_true = bandpass(y_true, 64, [self.min_bandpass_freq, self.max_bandpass_freq], self.bandpass_order)
            y_true = min_max_scale(y_true)
        
        return y_true, y_pred
    
    def get_predicted_peaks(self, signal):
        return get_peaks_for_hr(signal, 64, 3.0, -1, prominence = self.predicted_peaks_prominence, with_min_dist = True, with_valleys = False)
    
    def get_true_peaks(self, signal):
        return get_peaks_for_hr(signal, 64, 3.0, -1, prominence = self.true_peaks_prominence, with_min_dist = True, with_valleys = False)

    def prepare_dataset_from_subjects(self, truths, data_beg = 1000, data_end = 2000):
        """
        Prepare dataset from the given list of subjects.
        """

        data_arr = []
        for i in range(len(truths)):    
    
            truth = truths[i]
            data = truth.prepare_data_for_ml(self.num_feats_per_channel, self.skip_amount)
            data = data.iloc[data_beg: data_end, :]
            data['subject'] = i + 1
            data_arr.append(data)
        
        return pd.concat(data_arr)
