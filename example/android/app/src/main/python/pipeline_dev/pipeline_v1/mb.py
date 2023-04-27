"""
Wrapper class for the XGBoost wrapper, called MoodBoost since it's for Mood Triggers.

April 27, 2023
"""

import numpy as np
import xgboost as xgb

from losses import LossFactory
    

class LonePineGBM:
    
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
        self.num_leaves = num_leaves
        self.max_bin = max_bin

        # hyperparams for memory features from raw RGB
        self.num_feats_per_channel = num_feats_per_channel
        self.skip_amount = skip_amount

        self.model = None
        self.training_loss = None
        self.test_loss = None

        # process IEEE data; exclude subject if specified
        self.given_data = self.prepare_dataset_from_subjects(truths, data_beg = data_beg, data_end = data_end)
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
        for batch_num in range(batches):
            
            # curr batch is a random sample of the training splits
            batch_split_idxs = random.sample(range(len(self.train_splits)), batch_size)
            
            # ramdomly selected splits go in the batch and the rest stay in the training set
            batch_splits = [self.train_splits[i] for i in batch_split_idxs]
            self.train_splits = [self.train_splits[i] for i in range(len(self.train_splits)) if i not in batch_split_idxs]

            # create batch data and label
            batch_indices = [idx for split in batch_splits for idx in split]
            batch_rows = self.given_data.iloc[train_indices].drop(columns = [self.subject_col])
            batch_X = training_rows.drop(columns = [self.label_col]).to_numpy()
            batch_y = training_rows[self.label_col].to_numpy()

            # create XGBoost DMatrix for batch data and add the list of training sets
            batch_data = xgb.DMatrix(batch_X, batch_y)
            self.train_data.append(batch_data)

        # create the testing dataset
        test_indices = [idx for split in self.test_splits for idx in split]
        test_rows = self.given_data.iloc[test_indices].drop(columns = [self.subject_col])
        self.test_X = test_rows.drop(columns = [self.label_col]).to_numpy()
        self.test_y = test_rows[self.label_col].to_numpy()
        self.test_data = xgb.DMatrix(self.test_X, self.test_y)

        # initialize the loss function for the model
        self.loss = LossFactory(self.split_size, loss_type = loss_type, mse_weight = mse_weight, dtw_weight = dtw_weight).get_function()
    
    def split_data(self, to_exclude = None):
        
        data_in_use = self.given_data if to_exclude is None else self.given_data[~self.given_data[self.subject_col].isin(to_exclude)]

        subject_indices = data_in_use.groupby(self.subject_col).indices
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
        t1 = datetime.today()
        
        self.params = {
            'metric': 'None',
            'verbosity': -1,
            'learning_rate': self.learning_rate,
            'objective': 'regression',
            'boosting': self.model_type,
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'max_bin': self.max_bin,
        }
    
        if self.model_type == 'rf':
            self.params['bagging_freq'] = 1
            self.params['bagging_fraction'] = 0.8


        training_loss_key = 'hr_err'
        feval = self.hr_error_eval_metric
        print('loss is loss')
        
        training_meta = {}

        for train_data in self.train_data:
            
            if self.model_type == 'gbdt':
                self.gbm = lgb.train(
                    self.params,
                    train_data,
                    valid_sets = [train_data, self.test_data],
                    valid_names=['train', 'test'],
                    fobj = self.loss,
                    num_boost_round = self.n_estimators,
                    feval=feval,
                    callbacks=[
                        early_stopping(stopping_rounds = self.early_stopping_rounds),
                        log_evaluation(period=5)
                    ],
                    evals_result = training_meta,
                    init_model = self.gbm
                )
            else:
                self.gbm = lgb.train(
                    self.params,
                    train_data,
                    valid_sets = [train_data, self.test_data],
                    valid_names=['train', 'test'],
                    num_boost_round = self.n_estimators,
                    feval=feval,
                    callbacks=[
                        early_stopping(stopping_rounds = self.early_stopping_rounds),
                        log_evaluation(period=5)
                    ],
                    evals_result = training_meta,
                )

            mse, hr_err, hr_err_sq = self.eval()
            print(f'Before fine-tuning: MSE = {mse}, HR error = {hr_err}, HR error (squared) = {hr_err_sq}')

            if self.model_type == 'gbdt' and self.finetune:
                
                print('\n\nFine-tuning...')
                gbm_copy = copy.deepcopy(self.gbm)
                pred = gbm_copy.predict(train_data.get_data())
                
                # new_targ = train_data.get_label() - pred
                new_targ = np.ones(len(pred))
                nsplits = len(pred) // self.split_size
                labels = train_data.get_label()
                for i in range(nsplits):
                    pred_curr = pred[i * self.split_size: (i + 1) * self.split_size]
                    label_curr = labels[i * self.split_size: (i + 1) * self.split_size]
                    hr_err = self.get_hr_error(pred_curr, label_curr, square = True)
                    new_targ[i * self.split_size: (i + 1) * self.split_size] = hr_err
                
                new_train_data = lgb.Dataset(train_data.get_data(), label = new_targ)

                self.gbm = lgb.train(
                    self.params,
                    new_train_data,
                    valid_sets = [new_train_data, self.test_data],
                    valid_names=['train', 'test'],
                    fobj = self.loss,
                    num_boost_round = self.n_estimators // 2,
                    feval=feval,
                    callbacks=[
                        early_stopping(stopping_rounds = self.early_stopping_rounds // 2),
                        log_evaluation(period=5)
                    ],
                    evals_result = training_meta,
                    init_model = gbm_copy
                )

            

        self.training_loss = training_meta['train'][training_loss_key]
        self.test_loss = training_meta['test'][training_loss_key]
        print(f'Finished training in {datetime.today() - t1}')
    
    def fit_xgb(self):
        t1 = datetime.today()

        self.params = {
            'learning_rate': self.learning_rate,
            'booster': 'gbtree',
            'max_depth': self.max_depth,
            'num_leaves': self.num_leaves,
            'max_bin': self.max_bin,
        }

        feval = self.hr_error_eval_metric_xgb

        for train_data, train_data_just_data in zip(self.train_data, self.train_data_just_data):

            self.gbm = xgb.train(
                self.params,
                train_data,
                num_boost_round=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                feval=feval,
                verbose_eval=5,
                evals=[(train_data, 'train'), (self.test_data, 'test')],
                xgb_model=self.gbm,
                obj = self.loss.get_func()
            )

            # mse, hr_err, hr_err_sq = self.eval()
            # print(f'Before fine-tuning: MSE = {mse}, HR error = {hr_err}, HR error (squared) = {hr_err_sq}')

            if self.finetune:

                print('\n\nFine-tuning...')
                gbm_copy = self.gbm.copy()
                pred = gbm_copy.predict(train_data)

                new_targ = np.ones(len(pred))
                nsplits = len(pred) // self.split_size
                labels = train_data.get_label()
                for i in range(nsplits):
                    pred_curr = pred[i * self.split_size: (i + 1) * self.split_size]
                    label_curr = labels[i * self.split_size: (i + 1) * self.split_size]
                    new_targ[i * self.split_size: (i + 1) * self.split_size] = label_curr - pred_curr

                new_train_data = xgb.DMatrix(train_data_just_data, label=new_targ)

                self.gbm = xgb.train(
                    self.params,
                    new_train_data,
                    num_boost_round=self.n_estimators // 2,
                    early_stopping_rounds=self.early_stopping_rounds // 2,
                    feval=feval,
                    verbose_eval=5,
                    evals=[(new_train_data, 'train'), (self.test_data, 'test')],
                    xgb_model=gbm_copy,
                    obj = self.loss.get_func()
                )

        print(f'Finished training in {datetime.today() - t1}')

    def predict(self, X):
        return self.gbm.predict(X)
    
    def save(self, model_file = 'lonePineGBM.xgb'):

        # new_params = copy.deepcopy(self.params)
        # new_params['learning_rate'] = 0.0000001
        # new_gbm = lgb.train(self.params, self.train_data[0], num_boost_round=1, init_model = self.gbm)

        # # import onnxmltools
        # # from onnxconverter_common.data_types import FloatTensorType

        # # initial_types = [('input', FloatTensorType([None, new_gbm.num_feature()]))]
        # # onnx_model = onnxmltools.convert_lightgbm(new_gbm, initial_types=initial_types)
        # # onnxmltools.utils.save_model(onnx_model, model_file)

        # import onnxmltools
        # from onnxconverter_common.data_types import FloatTensorType
        # onnx_model = onnxmltools.convert_lightgbm(new_gbm, initial_types=[('input', FloatTensorType([None, new_gbm.num_feature()]))])

        # # Save as protobuf
        # onnxmltools.utils.save_model(onnx_model, model_file)

        self.gbm.save_model(model_file)

    
    def load_from_file(self, model_file):
        self.gbm = lgb.model_from_string(model_file)

    def eval(self):
        
        test_X = self.test_data.get_data()
        test_y = self.test_data.get_label()
        nsplits = int(len(test_X) / self.split_size)
        errs = []
        mses = np.zeros(len(test_X))
        
        for i in range(nsplits):

            curr_pred = self.predict(test_X[i * self.split_size: (i + 1) * self.split_size, :])
            curr_true = test_y[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 5, use_bandpass = True)
            
            mses[i * self.split_size: (i + 1) * self.split_size] = curr_true - curr_pred
            hr_err = self.get_hr_error(curr_true, curr_pred, square = False)
            errs.append(hr_err)
        
        return np.mean(np.square(mses)), np.mean(errs), np.mean(np.square(errs))

    def xgb_eval(self):
        
        nsplits = int(len(self.test_X) / self.split_size)
        errs = []
        mses = np.zeros(len(self.test_X))
        
        for i in range(nsplits):

            curr_pred = self.predict(xgb.DMatrix(self.test_X[i * self.split_size: (i + 1) * self.split_size, :]))
            curr_true = self.test_y[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 5, use_bandpass = True)
            
            mses[i * self.split_size: (i + 1) * self.split_size] = curr_true - curr_pred
            hr_err = self.get_hr_error(curr_true, curr_pred, square = False)
            errs.append(hr_err)
        
        return np.mean(np.square(mses)), np.mean(errs), np.mean(np.square(errs))
    
    def validate(self):

        test_X = self.test_data.get_data()
        test_y = self.test_data.get_label()
        nsplits = int(len(test_X) / self.split_size)
        
        errors = []
        for i in range(nsplits):

            curr_pred = self.predict(test_X[i * self.split_size: (i + 1) * self.split_size, :])
            curr_true = test_y[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 5, use_bandpass = True)
            
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

    def xgb_validate(self):

        test_X = self.test_X
        test_y = self.test_y
        nsplits = int(len(test_X) / self.split_size)
        
        errors = []
        for i in range(nsplits):
            curr_X = xgb.DMatrix(test_X[i * self.split_size: (i + 1) * self.split_size, :])
            curr_pred = self.predict(curr_X)
            curr_true = test_y[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 5, use_bandpass = True)
            
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

    def plot_loss(self):
        if self.training_loss is not None and self.test_loss is not None:
            training_loss_normed = min_max_scale(self.training_loss)
            test_loss_normed = min_max_scale(self.test_loss)
            plt.plot(training_loss_normed, label = 'training loss')
            plt.plot(test_loss_normed, label = 'test loss')
            plt.legend()
        
    def get_model_stats(self):

        model_info = self.gbm.dump_model()
        tree_depths = []

        for tree_info in model_info['tree_info']:
            tree_structure = tree_info['tree_structure']
            
            # Recursive function to compute the depth of a tree
            def calculate_depth(node, current_depth=0):
                if 'leaf_value' in node:
                    return current_depth
                else:
                    left_depth = calculate_depth(node['left_child'], current_depth + 1)
                    right_depth = calculate_depth(node['right_child'], current_depth + 1)
                    return max(left_depth, right_depth)

            tree_depth = calculate_depth(tree_structure)
            tree_depths.append(tree_depth)
        

        print(f'Best test loss: {min(self.test_loss)}\n')
        print('Tree depth stats:')
        print('Min tree depth:', min(tree_depths))
        print('Max tree depth:', max(tree_depths))
        print('Avg tree depth:', np.mean(tree_depths))
        print('\nFeature importances:')
        display(self.get_feature_importances())
    
    def get_feature_importances(self):
        importances = self.gbm.feature_importance(importance_type='gain')
        feature_importances = pd.DataFrame({'feature': self.features, 'importance': importances})
        feature_importances = feature_importances.sort_values('importance', ascending=False)
        return feature_importances
    
    def hr_error_eval_metric(self, y_pred, eval_data):
        y_true = eval_data.get_label()
        nsplits = int(len(y_pred) / self.split_size)
        hr_err = []
        for i in range(nsplits):
            curr_pred = y_pred[i * self.split_size: (i + 1) * self.split_size]
            curr_true = y_true[i * self.split_size: (i + 1) * self.split_size]
            curr_true, curr_pred = self.process_signal(curr_true, curr_pred, smoothing_window = 10, use_bandpass = True)
            hr_err.append(self.get_hr_error(curr_true, curr_pred, square = False))
        return 'hr_err', np.mean(hr_err), False
    
    def hr_error_eval_metric_xgb(self, y_pred, eval_data):
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
        true_peaks, _ = self.get_true_peaks(y_true)
        pred_peaks, _ = self.get_predicted_peaks(y_pred)
        if square:
            return np.power(len(true_peaks) - len(pred_peaks), 2)
        return abs(len(true_peaks) - len(pred_peaks))
    
    def get_hrv_error(self, y_true, y_pred, square = True):

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
    
        orig_len = len(y_pred)
        # y_pred = ppg_to_bvp(y_pred, 64)
        y_pred = n_moving_avg(y_pred, smoothing_window)
        y_pred = resample(y_pred, orig_len)
        if use_bandpass:
            y_pred = bandpass(y_pred, 64, [self.min_bandpass_freq, self.max_bandpass_freq], self.bandpass_order)
        y_pred = min_max_scale(y_pred)
        
        y_true = n_moving_avg(y_true, 20)
        y_true = resample(y_true, orig_len)
        if use_bandpass:
            y_true = bandpass(y_true, 64, [self.min_bandpass_freq, self.max_bandpass_freq], self.bandpass_order)
        y_true = min_max_scale(y_true)
        
        return y_true, y_pred
    
    def get_predicted_peaks(self, signal):
        return get_peaks_v2(signal, 64, 3.0, -1, prominence = self.predicted_peaks_prominence, with_min_dist = True, with_valleys = False)
    def get_true_peaks(self, signal):
        return get_peaks_v2(signal, 64, 3.0, -1, prominence = self.true_peaks_prominence, with_min_dist = True, with_valleys = False)

    def prepare_dataset_from_subjects(self, truths, data_beg = 1000, data_end = 2000):
        data_arr = []
        for i in range(len(truths)):    
            truth = truths[i]
            data = truth.prepare_data_for_ml(self.num_feats_per_channel, self.skip_amount)
            data = data.iloc[data_beg: data_end, :]
            data['subject'] = i + 1
            data_arr.append(data)
        return pd.concat(data_arr)
