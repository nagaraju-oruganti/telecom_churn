import os
import numpy as np
import pandas as pd
import math
from functools import partial
from scipy.optimize import fmin
from sklearn.metrics import accuracy_score, roc_auc_score
import random

import warnings
warnings.filterwarnings('ignore')

#### Optimize proba threshold
class OptimizeThresh:
    def __init__(self, cfg, source_dir, kinds = ['pca', 'scaled']):
        self.cfg = cfg
        self.source_dir = source_dir
        self.req_cols = ['id', 'kfold', 'churn_probability', 'prob', 'pred']
        self.kinds = kinds

    def proba_threshold(self, df):
        best_acc, best_th = 0, 0
        for th in [th * 0.001 for th in range(0, 1001)]:
            preds = np.where(df['prob'] >= th, 1, 0)
            acc = accuracy_score(y_true = df['churn_probability'], y_pred = preds)
            if acc > best_acc:
                best_acc = acc
                best_th = th
        
        ## optim preds
        df['optim_pred'] = np.where(df['prob'] >= best_th, 1, 0)
        acc = accuracy_score(y_true = df['churn_probability'], y_pred = df['optim_pred'])
        return df, acc, best_th
    
    def run(self):
        dirpath = f'{self.cfg.models_dir}/predictions/{self.source_dir}'
        ignore_models = []
        self.df = pd.DataFrame()
        dict_model_threshold = {k : {} for k in self.kinds}
        for source_kind in self.kinds:
            files = os.listdir(f'{dirpath}/{source_kind}')
            for ignore in ignore_models:
                files = [f for f in files if ignore not in f]
            
            # combine
            for f in files:
                clf_name = f.replace('all_folds_', '').replace('.csv', '')
                df = pd.read_csv(f'{dirpath}/{source_kind}/{f}', index_col = 0)
                df = df[self.req_cols]
                df, acc, th = self.proba_threshold(df)
                os.makedirs(f'{dirpath}/{source_kind}_optim_thresh', exist_ok = True)
                df.to_csv(f'{dirpath}/{source_kind}_optim_thresh/{f}')
                
                print(f'{source_kind:10s} {f:40s} with acc: {round(acc, 4)} and optim_threshold {round(th, 4)}')
                
                dict_model_threshold[source_kind][clf_name] = th
            print()
            
        return dict_model_threshold
                
#### OPTIMIZATION
class Optimize:
    def __init__(self):
        self.coef_ = 0
    
    def _optim_threshold(self, x_wt, y):
        best_acc, best_th = 0, 0
        for th in [th * 0.01 for th in range(0, 101)]:
            acc = accuracy_score(y_true = y, y_pred = np.where(x_wt >= th, 1, 0))
            if acc > best_acc:
                best_acc = acc
                best_th = th
        return best_th
    
    def _metric(self, coef, X, y):
        x_coef = X * coef
        x_wt = np.sum(x_coef, axis = 1)
        optim_th = self._optim_threshold(x_wt, y)
        predictions = np.where(x_wt > optim_th, 1, 0)
        acc_score = accuracy_score(y_true = y, y_pred = predictions)
        return -1.0 * acc_score

    def fit(self, X, y):
        partial_loss = partial(self._metric, X = X, y = y)
        init_coef = np.random.dirichlet(np.ones(X.shape[1]))
        self.coef_ = fmin(partial_loss, init_coef, disp = True)
        self.optim_th = self._optim_threshold(np.sum( X * self.coef_, axis = 1), y)
        print(f'optim_th: {self.optim_th}')
        
    def predict(self, X):
        x_coef = X * self.coef_
        predictions = np.where(np.sum(x_coef, axis = 1) > self.optim_th, 1, 0)
        return predictions

class Ensambler:
    def __init__(self, cfg, source_dir):
        self.cfg = cfg
        self.source_dir = source_dir
        self.req_cols = ['id', 'kfold', 'churn_probability', 'optim_pred']
        self.params_ = 0
        
    def load_predictions(self):
        dirpath = f'{self.cfg.models_dir}/predictions/{self.source_dir}'
        ignore_models = ['svm']
        self.df = pd.DataFrame()
        for source_kind in ['scaled']:
            files = [f for f in os.listdir(f'{dirpath}/{source_kind}_optim_thresh') if 'all_folds' in f]
            for ignore in ignore_models:
                files = [f for f in files if ignore not in f]
            
            # combine
            for f in files:
                clf_name = f.replace('all_folds_', '').replace('.csv', '')
                df = pd.read_csv(f'{dirpath}/{source_kind}_optim_thresh/{f}', index_col = 0)
                df = df[self.req_cols]
                df.rename(columns = {c : f'{source_kind}_{clf_name}_{c}' for c in ['optim_pred']}, inplace = True)
                self.df = df if self.df.empty else self.df.merge(df, on = ['id', 'kfold', 'churn_probability'], how = 'inner')
    
    def optimize_for_fold(self, fold):
        train_df = self.df[self.df['kfold'] != fold]
        valid_df = self.df[self.df['kfold'] == fold]
    
        xtrain = train_df[self.cols].values
        ytrain = train_df[self.cfg.target].values
        
        opt = Optimize()
        opt.fit(xtrain, ytrain)

        xvalid = valid_df[self.cols].values
        yvalid = valid_df[self.cfg.target].values
        preds = opt.predict(xvalid)
        acc = accuracy_score(yvalid, preds)
        auc = roc_auc_score(yvalid, preds)
        print(f'Accuracy score for fold {fold}: {acc} with AUC: {auc}')
        print('coefs:       ', opt.coef_)
        print('threshold:   ', opt.optim_th)
        
        return opt.coef_, acc, opt.optim_th
        
    def run_optimizer(self):
        self.cols = [c for c in self.df.columns if ('optim_pred' in c) & (c != self.cfg.target)]
        print('order of models:')
        print(self.cols)
        
        self.coefs, self.accuracies, self.threshs = [], [], []
        for fold in range(5):
            opt_coefs, acc,th = self.optimize_for_fold(fold)
            self.coefs.append(opt_coefs)
            self.accuracies.append(acc)
            self.threshs.append(th)
        
        self.coefs = np.array(self.coefs)
        self.mean_coefs = np.mean(self.coefs, axis = 0)
        print('coefs:       ', self.mean_coefs)
        print('accuracy:    ', np.mean(self.accuracies))
        print('threshold:   ', np.mean(self.threshs))
        
        ### Final
        X = self.df[self.cols].values
        y = self.df[self.cfg.target].values
        preds = np.where(np.sum(X * self.mean_coefs, axis = 1) > np.mean(self.threshs), 1, 0)
        accuracy = accuracy_score(y, preds)
        print(f'Final accuracy: {accuracy}')
        
        #### Heuristics for final
        dict_coef = {c:self.mean_coefs[idx] for idx, c in enumerate(self.cols)}
        return dict_coef, np.mean(self.threshs)
        
if __name__ == '__main__':
    from config import Config
    config = Config()
    config.models_dir = 'models'
    e = Ensambler(cfg = config, source_dir = 'baseline')
    e.load_predictions()
    e.run_optimizer()