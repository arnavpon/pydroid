"""
Optimizatin model with both the subject-wise cross-validation and Bayesian optimization.
"""

import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from pipeline_v1.mb import MoodBoost


HYPERPARAM_SPACE = [
    Integer(50, 300, name = "n_estimators"),
    Integer(640, 1280, name = "split_size"),
    Real(0.002, 0.5, name = "learning_rate"),
    Integer(10, 100, name = "early_stopping_rounds"),
    Real(0.0, 1.0, name = "mse_weight"),
    Real(0.0, 1.0, name = "dtw_weight"),
    Integer(1000, 4000, name = "data_beg"),
    Integer(6000, 10000, name = "data_end"),
    Integer(1, 8, name = "batches"),
    Real(0.4, 1.0, name = "min_bandpass_freq"),
    Real(2.5, 4.0, name = "max_bandpass_freq"),
    Integer(2, 6, name = "bandpass_order"),
    Real(0.1, 0.75, name = "predicted_peaks_prominence"),
    Real(0.1, 0.5, name = "true_peaks_prominence"),
    Integer(3, 10, name = "max_depth"),
    Integer(100, 300, name = "max_bin"),
    Integer(3, 10, name = "num_feats_per_channel"),
    Integer(5, 25, name = "skip_amount"),
]

                                          
def SubjectwiseCrossVal(truths, random_state = None, loss_type = 'combined',
        n_estimators = 100, split_size = 1280, learning_rate = 0.1, test_size = 0.3, early_stopping_rounds = 50,
        mse_weight = None, dtw_weight = None, data_beg = 1000, data_end = 10000, batches = 1, finetune = True, 
        min_bandpass_freq = 0.7, max_bandpass_freq = 4.0, bandpass_order = 4,
        predicted_peaks_prominence = 0.22, true_peaks_prominence = 0.15,
        max_depth = 7, max_bin = 255, num_feats_per_channel = 3, skip_amount = 15,
        rounds_per_model = 1, collect = False):
    """
    Subject-wise cross-validation. Performed by holding out each subject from training and then testing on that subject,
    and then averaging the results across all the subjects. Due to the randomness in the models, use the rounds_per_model
    param to run `rounds_per_model` validation runs for each subject. Use the collect param to collect the validation for each
    test metric for further analysis.

    NOTE: The truths argument are IeeeGroundTruth objects.

    EDIT: truths changed to now contain a dictionary where keys are the subject idx and values are a list of truths,
    corresponding to that subject's different augmented versions.
    """
    
    # dictionary holds each model trained, per subject
    models = {}
    
    for subj_to_exlude in truths:
        
        # init entry for this subj index
        models[subj_to_exlude] = []

        training_truths = []
        for subj in truths:
            if subj == subj_to_exlude:
                continue
            training_truths += truths[subj]['training']

        
        for _ in range(rounds_per_model):

            mod = MoodBoost(
                truths = training_truths,
                random_state = random_state,
                loss_type = loss_type,
                n_estimators = n_estimators,
                split_size = split_size,
                learning_rate = learning_rate,
                test_size = test_size,
                early_stopping_rounds = early_stopping_rounds,
                mse_weight = mse_weight,
                dtw_weight = dtw_weight,
                data_beg = data_beg,
                data_end = data_end,
                batches = batches,
                finetune = finetune,
                min_bandpass_freq = min_bandpass_freq,
                max_bandpass_freq = max_bandpass_freq,
                bandpass_order = bandpass_order,
                predicted_peaks_prominence = predicted_peaks_prominence,
                true_peaks_prominence = true_peaks_prominence,
                max_depth = max_depth,
                max_bin = max_bin,
                num_feats_per_channel = num_feats_per_channel,
                skip_amount = skip_amount,
            )
            
            mod.fit()
            models[subj_to_exlude].append(mod)
    
    # dictionary holds the validation results for each model, per subject
    model_performances = {}
    for subject in models:
        
        # track the model performances for this subject
        model_performances[subject] = {}

        # collect all the training and testing triths for this subject
        testing_truths = truths[subject]['training'] + truths[subject]['testing']
        
        for mod_num, mod in enumerate(models[subject]):
            model_performances[subject][mod_num] = []

            for testing_truth in testing_truths:
                
                truth_data = testing_truth.prepare_data_for_ml(8, 12)
                valX = truth_data.drop(columns = ['bvp']).to_numpy()
                valY = truth_data['bvp'].to_numpy()
                errs = mod.validate(val_data = (valX, valY))
                model_performances[subject][mod_num].append(errs)
            
    return model_performances



class MoodBoostOptimizer:
    """
    Bayesian optimization class for hyperparameter tuning.
    """

    def __init__(self, truths):
        """
        Just pass the necessary IeeGroundTruth objects to the model.
        """

        self.truths = truths
    
    def objective(self, n_estimators = 100, split_size = 1280, learning_rate = 0.1, early_stopping_rounds = 50,
                    mse_weight = None, dtw_weight = None, data_beg = 1000, data_end = 2000, batches = 1, finetune = True, 
                    min_bandpass_freq = 0.67, max_bandpass_freq = 3.0, bandpass_order = 4,
                    predicted_peaks_prominence = 0.22, true_peaks_prominence = 0.15,
                    max_depth = 7, max_bin = 255, num_feats_per_channel = 3, skip_amount = 15):
        """
        Computes a HR score for a combination of parameters using the cross-validation function.
        """

        # use try-except to catch combinations of params that cause errors and penalize them
        try:
            
            hr_score, _, _ = SubjectwiseCrossVal(
                self.truths,
                random_state = None,
                loss_type = 'combined',
                n_estimators = n_estimators,
                split_size = split_size,
                learning_rate = learning_rate,
                early_stopping_rounds = early_stopping_rounds,
                mse_weight = mse_weight,
                dtw_weight = dtw_weight,
                data_beg = data_beg,
                data_end = data_end,
                batches = batches,
                finetune = finetune,
                min_bandpass_freq = min_bandpass_freq,
                max_bandpass_freq = max_bandpass_freq,
                bandpass_order = bandpass_order,
                predicted_peaks_prominence = predicted_peaks_prominence,
                true_peaks_prominence = true_peaks_prominence,
                max_depth = max_depth,
                max_bin = max_bin,
                num_feats_per_channel = num_feats_per_channel,
                skip_amount = skip_amount,
            )
            return hr_score
        except:
            return 1000
    
    def optimize(self, n_calls = 50):
        """
        Hyperparam optimization, using the HYPERPARAM_SPACE array with the ranges of each
        param hardcoded.
        """

        @use_named_args(HYPERPARAM_SPACE)
        def wrapped_objective(**params):
            return self.objective(**params)
        
        result = gp_minimize(
            wrapped_objective, HYPERPARAM_SPACE, n_calls = n_calls, random_state = 42, verbose = 1
        )

        return result
