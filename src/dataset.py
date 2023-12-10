import os
import pandas as pd
from utils import create_folds

class Dataset:
    '''Prepare train and validation datasets
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.source_kind = 'scaled_train_folds.csv' if self.cfg.source_kind == 'scaled' else 'pca_train_folds.csv'
        self.test_source_kind = self.source_kind.replace('train', 'test').replace('_folds', '')
        if os.path.exists(f'{cfg.data_dir}/{self.source_kind}'):
            self.df = pd.read_csv(f'{cfg.data_dir}/{self.source_kind}', index_col = 0)
            self.test_df = pd.read_csv(f'{cfg.data_dir}/{self.test_source_kind}')
        else:
            self.df = create_folds(cfg.data_dir, k = 5)
            
    def split(self, fold):
        # There are few features that need to excluded from the modeling
        remove = self.cfg.features_not_considered
        self.df.drop(columns = [r for r in remove if r in self.df.columns], inplace = True)
        self.test_df.drop(columns = [r for r in remove if r in self.test_df.columns], inplace = True)
        self.test_df[self.cfg.target] = 0
        
        # Make split on fold
        train = self.df[self.df['kfold'] != fold]      # fold 0 (train 1, 2, 3, 4)
        valid = self.df[self.df['kfold'] == fold]      # fold 0 (valid 0)
        
        train.drop(columns = ['kfold'], inplace = True)
        valid.drop(columns = ['kfold'], inplace = True)
        
        return train, valid