###### INSTALL REQUIRED LIBRARIES
import sys
import subprocess
import pkg_resources
import importlib
def install_libraries(required_packages):
    '''Install required packages
    '''    
    for pckg in required_packages:
        if not importlib.util.find_spec(pckg):
            subprocess.run([sys.executable, '-m', 'pip', 'install', pckg, '-q'], check=True)

class Config:
    
    #### Packages to install
    ## pip install [package_name]
    required_packages = [
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'optuna',
    ]
    
    #### Data arguments
    data_dir = '../data'
    models_dir = '../models'
    source_kind = 'scaled'
    
    #### Features exclude from modeling
    features_not_considered =  ['circle_id']
    target = 'churn_probability'
    
    #### Train arguments
    folds = range(5)
    
    #### Random seed
    seed = 42
    
    #### model keyword parameters
    load_optim_params = False                   # if True, then the models are trained with optimium parameters estimated with Optuna
                                                # otherwise, default (below) parameters will be laoded.
    #--- predicted accuracy is wih split 0.2 on the entire dataset
    # Logistic regression
    lr_params = {
        'solver': 'lbfgs',
        'C': 1.0,
        'max_iter': 200,
        'class_weight': None,
        'random_state': seed
    }
    
    # KNN Parameters
    knn_params = {
        'n_neighbors': 3,
        'weights': 'distance',
        'algorithm': 'auto',
        'p': 1,
        'n_jobs': -1
    }
    
    # decision tree
    decision_tree_params = {
        'criterion' : 'gini',
        'max_depth' : 9,
        'max_features': 'log2',
        'random_state' : seed,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
    }
    
    # random forest
    random_forest_params = {
        'n_estimators' : 200,
        'criterion': 'entropy',
        'max_depth' : 3,
        'max_features': None,
        'random_state' : seed,
        'n_jobs' : -1
    }
    
    # support vector machine
    svm_params = {
        'kernel' : 'rbf',
        'gamma' : 'auto',
        'C' : 1,
        'max_iter': 1000,
        'probability' : True,
        'random_state': seed,
    }
    
    gradient_boosting_params = {
        'n_estimators' : 200,
        'learning_rate' : 0.1,
        'max_depth' : 3,
        'max_features' : None,
        'min_samples_leaf' : 1,
        'min_samples_split' : 2,
        'random_state' : seed,
    }
    
    # xgboost
    xgboost_params = {
        'objective'         : 'binary:logistic',
        'tree_method'       : 'hist', 
        'n_estimators'      : 500,
        'max_depth'         : 10,
        'learning_rate'     : 0.1,
        'min_samples_split' : 10,
        'random_state'      : seed,
        'n_jobs': -1
    }
    
    ### ALLOW REPORTING RESULTS
    surpress_reporting = False
    save_predictions = True
    
    #### OPTUNA
    override_previous_best = False