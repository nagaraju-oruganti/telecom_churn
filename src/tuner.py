import os
import time
import pandas as pd
import numpy as np

## LIBRARIES FOR MODELING - ensure you install them
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

## LOAD LOCAL PACKAGES
from config import Config
from dataset import Dataset
import utils
from utils import (save_best_parameters,
                   EarlyStoppingExceeded, 
                   early_stopping_opt,
                   OPTUNA_EARLY_STOPING,
                   class_weight)

## OPTUNA for parameter tuning
import optuna

## Surpress warnings
import warnings
warnings.filterwarnings('ignore')

################################################################
### CLASSFICATION MODELS
################################################################
class ParametersTuner:
    def __init__(self,cfg):
        self.cfg = cfg
    
    # SAVE PREDICTIONS   
    def __save_results(self, cls_name, result):
        # save parameters
        save_best_parameters(self.cfg, cls_name, result, overide=self.cfg.override_previous_best)
        print(f'Best parameters for the {cls_name} is saved')
    
    # Construct dataset
    def __make_data(self, fold):
        train, valid = Dataset(self.cfg).split(fold) 
        X_train = train.drop(columns = [self.cfg.target, 'id'])
        y_train = train[self.cfg.target]
        
        X_valid = valid.drop(columns = [self.cfg.target, 'id'])
        y_valid = valid[self.cfg.target]
        return X_train, y_train, X_valid, y_valid
        
    ##### OBJECTIVE FUNCTIONS
    def objective_knn(self, trial):
        from sklearn.neighbors import DistanceMetric
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        params = {
            'n_neighbors'   : trial.suggest_categorical('n_neighbors', range(2, 10)),
            'algorithm'     : trial.suggest_categorical('algorithm', ['ball_tree', 'kd_tree', 'brute', 'auto']),
            'weights'       : trial.suggest_categorical('weights', ['distance', 'uniform']),
            'p'             : trial.suggest_categorical('p', [1, 2]),
            'metric'        : trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'l2', 'l1', 'cityblock']),
            'n_jobs'        : -1,
        }
        
        acc = []
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            clf = KNeighborsClassifier(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
        
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    def objective_lr(self, trial):
        
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        acc = []
        params = {
            'solver'        : trial.suggest_categorical('solver',           ['lbfgs', 'liblinear', 'newton-cg']),
            'C'             : trial.suggest_categorical('C',                [100, 10, 1, 0.5, 0.1]),
            'max_iter'      : trial.suggest_int(        'n_estimators',     low = 200, high = 700, step = 50),
            'random_state'  : trial.suggest_categorical('random_state',     [42, 3407]),
            'n_jobs'        : -1,
        }
        
        # applyp params over folds
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            dict_samples = dict(y_train.value_counts())
            params.update({'class_weight': class_weight(dict_samples)})
            
            clf = LogisticRegression(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
            
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    def objective_dtree(self, trial):
        
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        acc = []
        max_depth = [None] + list(range(1, 41, 4))
        params = {
            'criterion'         : trial.suggest_categorical('criterion',    ['entropy','gini', 'log_loss']),
            'max_depth'         : trial.suggest_categorical('max_depth',    max_depth),
            'max_features'      : trial.suggest_categorical('max_Features', ['sqrt', 'log2', None]),
            'min_samples_split' : trial.suggest_categorical('min_samples_split', [2, 4, 6, 8, 10]),
            'min_samples_leaf'  : trial.suggest_categorical('min_samples_leaf', [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            'random_state'      : trial.suggest_categorical('random_state', [42, 3407]),
        }
        
        # applyp params over folds
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            dict_samples = dict(y_train.value_counts())
            params.update({'class_weight': class_weight(dict_samples)})
            
            clf = DecisionTreeClassifier(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
            
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    def objective_rf(self, trial):
        
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        acc = []
        params = {
            'n_estimators'  : trial.suggest_int(        'n_estimators', low = 50, high = 1000, step = 20),
            'criterion'     : trial.suggest_categorical('criterion',    ['entropy','gini', 'log_loss']),
            'max_depth'     : trial.suggest_categorical('max_depth',    [ None, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39]),
            'max_features'  : trial.suggest_categorical('max_Features', ['sqrt', 'log2', None]),
            'random_state'  : trial.suggest_categorical('random_state', [42, 3407]),
            'n_jobs'        : -1,
        }
        
        # applyp params over folds
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            dict_samples = dict(y_train.value_counts())
            params.update({'class_weight': class_weight(dict_samples)})
            
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
            
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    
    def objective_svm(self, trial):
        
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        acc = []
        params = {         
            'kernel'        : trial.suggest_categorical('kernel',       ['linear', 'rbf', 'sigmoid']),
            'gamma'         : trial.suggest_categorical('gamma',        ['scale', 'auto']),
            'C'             : trial.suggest_categorical('C',            [100, 10, 1, 0.5, 0.1, 0.01]),
            'max_iter'      : trial.suggest_int(        'max_iter',     low = -1, high = 300, step = 20),
            'probability'   : True,
            'random_state'  : trial.suggest_categorical('random_state', [42, 3407]),
        }
        
        # applyp params over folds
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            dict_samples = dict(y_train.value_counts())
            params.update({'class_weight': class_weight(dict_samples)})
            
            clf = svm.SVC(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
            
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    def objective_xgboost(self, trial):
        
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        acc = []
        params = {
            'objective'         : 'binary:logistic',
            'n_estimators'      : trial.suggest_int(        'n_estimators',     low = 200, high = 800, step = 100),
            'max_depth'         : trial.suggest_categorical('max_depth',        [ None, 8, 12, 16]),
            #'min_samples_split' : trial.suggest_int(        'min_samples_split',low = 8, high = 16, step = 4),
            'learning_rate'     : trial.suggest_float(      'learning_rate',    0.1, 0.2),
            'random_state'      : trial.suggest_categorical('random_state',     [42, 3407]),
        }
        
        # applyp params over folds
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            dict_samples = dict(y_train.value_counts())
            params.update({'scale_pos_weight' : dict_samples[0]/dict_samples[1], 
                           'n_jobs': -1,
                           'min_samples_split': 10,
                           'tree_method': 'hist'})
            
            clf = XGBClassifier(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
            
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    def objective_gradient_boosting(self, trial):
        
        datasets = {fold : self.__make_data(fold) for fold in self.cfg.folds}
        
        acc = []
        params = {
            'n_estimators'      : trial.suggest_int(        'n_estimators',     low = 10, high = 1000, step = 50),
            'subsample'         : trial.suggest_float(      'subsample',        0.5, 1.0),
            'min_samples_split' : trial.suggest_int(        'min_samples_split',low = 2, high = 20, step = 2),
            'max_depth'         : trial.suggest_categorical('max_depth',        [ None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]),
            'max_features'      : trial.suggest_categorical('max_features',     [None, 'sqrt', 'log2']),
            'learning_rate'     : trial.suggest_float(      'learning_rate',    1e-3, 0.1),
            'random_state'      : trial.suggest_categorical('random_state',     [42, 3407]),
        }
        params.update({'loss' : 'log_loss'})
        
        # applyp params over folds
        for fold in self.cfg.folds:
            X_train, y_train, X_valid, y_valid = datasets[fold]
            dict_samples = dict(y_train.value_counts())
            
            clf = GradientBoostingClassifier(**params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            acc.append(accuracy_score(y_valid, y_valid_pred))
            
        # Track best model
        trial.set_user_attr(key= 'best_clf', value = clf.get_params())
        
        return np.mean(acc)
    
    ### OPTUNA Callback
    def __callback(self, study, trial):
        if self.study.best_trial.number == trial.number:
            self.study.set_user_attr(key='best_clf', value=trial.user_attrs['best_clf'])
    
    ### MODEL TUNERs
    def __tune(self, cls_name, objective_fn):
        
        print(f'===== Parameter tunner for {cls_name.upper()} started ===========================')
        self.study = optuna.create_study(direction = 'maximize')
        try:
            self.study.optimize(objective_fn, n_trials = 15, callbacks=[self.__callback, early_stopping_opt], n_jobs = -1)
        except EarlyStoppingExceeded:
            print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPING}')
            
        result = {
            'best_model_params'         : self.study.user_attrs['best_clf'],
            'best_search_space_params'  : self.study.best_params,
            'best_acc'                  : self.study.best_value
        }
        
        self.__save_results(cls_name = cls_name, result = result)
        
    ### MAIN TURER  
    def tune(self, skip_if_exists = False):
        if not self.cfg.classifiers:
            print('No base algorithm selected')
            
        if skip_if_exists:
            opt_dir = f'{self.cfg.models_dir}/optimization/{self.cfg.source_kind}'
            self.cfg.classifiers = [f for f in self.cfg.classifiers if f'{f}.pkl' not in os.listdir(opt_dir)]
            if not self.cfg.classifiers:
                print('No base algorithms identfied for parameters tuning.')
                return
        
        obj_mapper = {
            'knn'                   : self.objective_knn,
            'logistic_regression'   : self.objective_lr,
            'decision_tree'         : self.objective_dtree,
            'random_forest'         : self.objective_rf,
            'svm'                   : self.objective_svm,
            'gradient_boosting'     : self.objective_gradient_boosting,
            'xgboost'               : self.objective_xgboost,
        }
          
        for cls_name in self.cfg.classifiers:
            if cls_name in obj_mapper:
                self.__tune(cls_name=cls_name, objective_fn= obj_mapper[cls_name])
