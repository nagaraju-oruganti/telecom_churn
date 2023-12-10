### IMPORT LIBRARIES
import os
import pandas as pd
import pickle
import optuna

###### CREATE FOLDS
def create_folds(data_dir, k = 5):
    from sklearn.model_selection import KFold

    '''Create folds on target variable (churn_probability)

        Use the feature engineered dataset and split it into (k) parts where each
        get equal proportion of target labels.
        
        Parameters:
        - data_dir: path to the data directory (passed from configuration file)
        - k: # of split you will wish to make (default set to 5)
    '''
    
    ### Step 1: Load feature engineered dataset if exists in the directory, otherwise throw exception
    datapath = f'{data_dir}/scaled_train.csv'
    assert os.path.exists(datapath), Exception('Feature engineering dataset is not preset in the data directory. Perform feature engineering from the notebook.')
    df = pd.read_csv(datapath)
    df.reset_index(drop = True, inplace = True)
    
    ### Create split
    df['kfold'] = -1
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    X = df.drop(columns = ['churn_probability'])
    y = df[['churn_probability']]
    
    # Iterate over the folds
    for i, (t_idx, v_idx) in enumerate(kf.split(X = X, y = y)):
        df.loc[v_idx, ['kfold']] = i
    
    print('Samples in each fold:')
    print(df.kfold.value_counts())
    
    # save
    df.to_csv(f'{data_dir}/scaled_train_folds.csv')
    
    # Map the folds to PCA dataset
    pca_df = pd.read_csv(f'{data_dir}/pca_train.csv')
    pca_df = pca_df.merge(df[['id', 'kfold']], on = 'id')
    pca_df.to_csv(f'{data_dir}/pca_train_folds.csv')
    
    return df

### SAVE PREDICTIONS
def save_predictions_single_fold(train_df, test_df, clf, y_train_proba, y_train_pred, 
                                 y_valid_proba, y_valid_pred, 
                                 classes,
                                 dest_name, cls_name):
    output_dir = f'{clf.models_dir}/predictions/{dest_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    # We will save probablity of class label 1 only [prob(class 0) = 1 - prob(class 1)]
    idx = list(classes).index(1)
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['kfold'] = 'train'
    train = train_df[['id', 'churn_probability', 'kfold']]
    train['prob'] = y_train_proba[:, idx]
    train['pred'] = y_train_pred
    
    test_df['kfold'] = 'test'
    valid = test_df[['id', 'churn_probability', 'kfold']]
    valid['prob'] = y_valid_proba[:, idx]
    valid['pred'] = y_valid_pred
    
    df_pred = pd.concat([train, valid], axis = 0)
    df_pred.to_csv(f'{output_dir}/final_{cls_name}.csv')
    
    print(f'The predictions are saved for {cls_name} model...')
    
def save_predictions_all_folds(df, clf, results, classes, dest_name, cls_name):
    ### Note that results is a dictionary with keys fold, and subkeys proba and pred
    output_dir = f'{clf.models_dir}/predictions/{dest_name}'
    os.makedirs(output_dir, exist_ok=True)
    # While feature engineering, we mapped churn_probability labels as {'5-Order': 1, 'Lost' : 0}
    # We will save probablity of class label 1 only [prob(class 0) = 1 - prob(class 1)]
    idx = list(classes).index(1)
    dfs = []
    for fold, predictions in results[cls_name].items():
        valid = df[df['kfold'] == fold][['id', 'churn_probability', 'kfold']]
        valid['prob'] = predictions['proba'][:, idx]
        valid['pred'] = predictions['pred']
        dfs.append(valid)
    df_pred = pd.concat(dfs, axis = 0)
    df_pred.to_csv(f'{output_dir}/all_folds_{cls_name}.csv')
    print(f'The predictions are saved for {cls_name} model... {df_pred.shape}')
    
#### SAVE BEST PARAMETERS AFTER TUNING WITH OPTUNA
def save_best_parameters(cfg, cls_name, result, overide = False):
    best_path = f'{cfg.models_dir}/optimization/{cfg.source_kind}'
    os.makedirs(best_path, exist_ok=True)
    best_path = f'{best_path}/{cls_name}.pkl'
    
    if not overide:
        if os.path.exists(best_path):
            # check the previous saved score is better
            with open(best_path, 'rb') as file:
                saved_results = pickle.load(file)
                saved_acc = saved_results['best_acc']
                if saved_acc > result['best_acc']:
                    print('Accuracy saved from the previous run is better than the present. Present is discarded.')
                    print('Note: If you want to overide, then set the argument `overide` to True')
                    return
    # save
    with open(best_path, 'wb') as file:
        pickle.dump(result, file)

    ### END SAVED
    

#### OPTUNA CALLBACK FOR EARLY STOPPING
OPTUNA_EARLY_STOPING = 25
class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    if EarlyStoppingExceeded.best_score == None:
      EarlyStoppingExceeded.best_score = study.best_value

    if study.best_value < EarlyStoppingExceeded.best_score:
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
      if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            best_score = None
            raise EarlyStoppingExceeded()
      else:
            EarlyStoppingExceeded.early_stop_count=EarlyStoppingExceeded.early_stop_count+1
    #print(f'EarlyStop counter: {EarlyStoppingExceeded.early_stop_count}, Best score: {study.best_value} and {EarlyStoppingExceeded.best_score}')
    return

#### CLASS WEIGHTS
import numpy as np
def class_weight(labels_dict, mu=0.15):
    total = sum(labels_dict.values())
    keys = labels_dict.keys()
    weights = dict()
    for i in keys:
        score = np.log((mu*total)/float(labels_dict[i]))
        weights[i] = score if score > 1 else 1
    return weights

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

if __name__ == '__main__':
    create_folds(data_dir ='../data', k = 5)