import os
import time
import pandas as pd
import numpy as np
import pickle

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
from utils import (save_predictions_single_fold, 
                   save_predictions_all_folds, 
                   class_weight)

## Surpress warnings
import warnings
warnings.filterwarnings('ignore')

################################################################
### CLASSFICATION MODELS
################################################################
class ClassifierModels:
    def __init__(self,cfg):
        self.cfg = cfg
        self.results = {}
    
    # SPLIT DATASET ON FOLD
    def split(self, fold):
        # split dataset into train and test
        mdataset = Dataset(self.cfg)
        self.df = mdataset.df
        self.test_df = mdataset.test_df
        
        if self.cfg.folds[0] == -1:
            print('Training for final model with all train samples')
            _ = mdataset.split(fold)
            self.df_train, self.df_valid = self.df.copy(), self.test_df.copy()
            self.df_train.drop(columns = ['kfold'], inplace = True)
        else:
            self.df_train, self.df_valid = mdataset.split(fold)
            
        # Drop kfold and churn_probability (target) columns to create features 
        # and select churn_probability column for target
        self.X_train = self.df_train.drop(columns = [self.cfg.target, 'id'])
        self.y_train = self.df_train[self.cfg.target] 
    
        self.X_valid = self.df_valid.drop(columns = [self.cfg.target, 'id'])
        self.y_valid = self.df_valid[self.cfg.target]
    
    # SAVE PREDICTIONS   
    def __save_predictions(self, cls_name):
        if self.cfg.save_predictions:
            dest_name = timestamp if self.cfg.dest_name == '' else self.cfg.dest_name
            if len(self.cfg.folds) > 1:
                save_predictions_all_folds(self.df, self.cfg, 
                                           self.results, 
                                           self.clf.classes_, dest_name, cls_name)
            if len(self.cfg.folds) == 1:
                save_predictions_single_fold(self.df_train, self.df_valid, self.cfg,
                                             self.y_train_proba, self.y_train_pred, 
                                             self.y_valid_proba, self.y_valid_pred, 
                                             self.clf.classes_, dest_name, cls_name)
                
        if len(self.cfg.folds) > 1:
            # report average and std of accuracy
            t_acc = [self.results[cls_name][fold]['train_acc'] for fold in self.cfg.folds]
            v_acc = [self.results[cls_name][fold]['valid_acc'] for fold in self.cfg.folds]
            
            mu_train, std_train = np.mean(t_acc), np.std(t_acc)
            mu_valid, std_valid = np.mean(v_acc), np.std(v_acc)
            
            print(f'Train accuracy scores - mean: {mu_train:.6f}, std: {std_train:.6f}')
            print(f'Valid accuracy scores - mean: {mu_valid:.6f}, std: {std_valid:.6f}')
            
        print('Train and prediction is complete.')
        
    def __load_optim(self, cls_name):
        filepath = f'{self.cfg.models_dir}/optimization/{self.cfg.source_kind}/{cls_name}.pkl'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as file:
                params = pickle.load(file)['best_model_params']
                return params
        return None
    
    ##### TRAINING ALGORITHMS    
    def __trainer(self, classifier, cls_name, set_params):
        print(f'===== TRAINING {cls_name.upper()} ===============================================')
        params = set_params
        if self.cfg.load_optim_params:
            optim_params = self.__load_optim(cls_name)
            if optim_params:
                params = optim_params
                print('Training with optimum parameters')

        for fold in self.cfg.folds:
            self.split(fold)
            dict_samples = dict(self.y_train.value_counts())
            if cls_name in ['knn', 'gradient_boosting']:
                ''
            else:
                if cls_name == 'xgboost':
                    params.update({'scale_pos_weight' : dict_samples[0]/dict_samples[1]})
                else:
                    params.update({'class_weight': class_weight(dict_samples)})
            self.clf = classifier(**params)                 # classifier initialization
            self.clf.fit(self.X_train, self.y_train)        # training part
            self.predict(cls_name, fold)                    # prediction (out-of-sample)    
                
        # save predictions
        self.__save_predictions(cls_name)
         
    def fit_predict(self):
        if not self.cfg.classifiers:
            print('No base algorithm selected')
            
        if len(self.cfg.folds) > 1:
            self.cfg.surpress_reporting = True
            
        params_mapper = {
            'knn'                   : (KNeighborsClassifier,        self.cfg.knn_params),
            'logistic_regression'   : (LogisticRegression,          self.cfg.lr_params),
            'decision_tree'         : (DecisionTreeClassifier,      self.cfg.decision_tree_params),
            'random_forest'         : (RandomForestClassifier,      self.cfg.random_forest_params),
            'gradient_boosting'     : (GradientBoostingClassifier,  self.cfg.gradient_boosting_params),
            'svm'                   : (svm.SVC,                     self.cfg.svm_params),
            'xgboost'               : (XGBClassifier,               self.cfg.xgboost_params),
        }
        
        for cls_name in self.cfg.classifiers:
            if cls_name in params_mapper:
                classifier, params = params_mapper[cls_name]
                self.__trainer(classifier, cls_name, params)
            
    def predict(self, cls_name, fold):
        # prediction on train dataset
        self.y_train_pred = self.clf.predict(self.X_train)
        self.y_train_proba = self.clf.predict_proba(self.X_train)
        
        # prediction on test dataset
        self.y_valid_pred = self.clf.predict(self.X_valid)
        self.y_valid_proba = self.clf.predict_proba(self.X_valid)
        
        ## estimate accuracy for both train and test
        train_accuracy = accuracy_score(self.y_train, self.y_train_pred)
        valid_accuracy = accuracy_score(self.y_valid, self.y_valid_pred)
        
        if cls_name not in self.results:
            self.results[cls_name] = {fold: {} for fold in self.cfg.folds}
        
        self.results[cls_name][fold] = {
            'proba'    : self.y_valid_proba,
            'pred'     : self.y_valid_pred,
            'train_acc': train_accuracy,
            'valid_acc': valid_accuracy,
            }
    
        ## repost the findings
        if not self.cfg.surpress_reporting:
            print(f'- train accuracy: {round(train_accuracy, 5) * 100 }')
            print(f'- valid accuracy: {round(valid_accuracy, 5) * 100 }')
            print('')
            