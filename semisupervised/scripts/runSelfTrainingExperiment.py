import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

from random import sample

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

class runExperiment:
    """
    
    This class trains and evaluates several different supervised classification
    models, and their semi supervised extensions, on a given dataset where 
    each of 10%, 20%, ..., 90% of the labels are intentionally removed.
    
    Parameters:
    -----------
    n_splits: int
        Number of stratified samples. Each model, at each percentage of missing
        labels, is trained individually on each sample.
    
    test_size: float
        Size (percentage of total) of the test set. Should be in (0,1).
    
    f1_average_param: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
        The 'average' parameter of sklearn.metrics.f1_score.
        
    
    lo_unlabeled_pct: list(float):
        The percentages of missing data to train each model on. Each element 
        should be in (0,1).
    
    Attributes:
    ----------
    n_splits:
        as initalized.
    
    test_size:
        as initalized
        
    f1_average_param:
        as initalized
        
    lo_unlabeled_pct:
        as initalized
    
    results: pandas dataframe
        The results of the experiment. Each row corresponds to a specific sample
       (at a given percentage of unlabeled data). The columns contain the 
       corresponding f1 score/f1 score ratio of the models. There a total of len(lo_unlabeled_pct) * n_splits
       rows.
       
    agg_results: pandas dataframe
        The results dataframe grouped by unlabeled percent. Each row corresponds
        to the percentage of unlabeled data. The columns contain the mean,
        std, 97.5% and 2.5% quantiles of the f1 score/f1 score ratio of the models.
    """
    
    def __init__(self, n_splits, test_size, f1_average_param,
                 lo_unlabeled_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.f1_average_param = f1_average_param
        self.lo_unlabeled_pct = lo_unlabeled_pct    
        
    def fit(self, X, y):
        """
        Runs the experiment on X and  y.
        
        Parameters
        ---------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
            
        y: {array-like, sparse matrix} of shape (n_samples,)
            Array representing the labels.

        Returns
        -------
        None
        """        
        
        n_samples = len(y)        
        self.results = pd.DataFrame()
        
        #k in kbest is set to 2.5% of the sample size so that it scales across datasets
        do_models = {
            'LR': LogisticRegression(max_iter = 500),
            
            'RF': RandomForestClassifier(warm_start = False),
            
            'KNN': KNeighborsClassifier(),
            
            'STC-T-LR': SelfTrainingClassifier(
                LogisticRegression(max_iter = 500)),
            
            'STC-T-RF': SelfTrainingClassifier(
                RandomForestClassifier(warm_start = False)),
            
            'STC-T-KNN': SelfTrainingClassifier(
                KNeighborsClassifier()),
            
            'STC-K-LR': SelfTrainingClassifier(
                criterion = 'k_best', k_best = int(0.025 * n_samples),
                base_estimator = LogisticRegression(max_iter = 500)),
            
            'STC-K-RF': SelfTrainingClassifier(
                criterion = 'k_best', k_best = int(0.025 * n_samples),
                base_estimator = RandomForestClassifier(warm_start = False)),
            
            'STC-K-KNN': SelfTrainingClassifier(
                criterion = 'k_best', k_best = int(0.025 * n_samples),
                base_estimator = KNeighborsClassifier())
            }
        
        #using shuffle split since test size needs to be sufficient for f1 score to be meaningful
        kf = StratifiedShuffleSplit(n_splits = self.n_splits,
                                    test_size = self.test_size)
        
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
                

            for unlabeled_pct in self.lo_unlabeled_pct:
                y_train_unlab = y_train.copy()   
                y_train_unlab[sample(list(y_train_unlab.index),
                                     k = int(unlabeled_pct * len(y_train_unlab)))] = -1

                do_scores = {'Unlabeled Percent': [unlabeled_pct]}                                                                     
                
                for model_name in do_models.keys():
                    if model_name[0:3] == 'STC':                   
                        do_models[model_name].\
                            fit(X_train, y_train_unlab)
                    else:
                        do_models[model_name].\
                            fit(X_train[y_train_unlab != - 1],
                                y_train_unlab[y_train_unlab != - 1])
                          
                    yhat = do_models[model_name].predict(X_test)
                    f1 = f1_score(y_test, yhat, average = self.f1_average_param)
                    do_scores[model_name] = [f1]

                self.results = pd.concat([self.results, pd.DataFrame(do_scores)])
                
        #for the STCs we replace their f1 scores ratios
        for model_name in do_models.keys():
            if model_name[0:3] == 'STC':
                self.results[model_name] /= self.results[model_name[6:]]
                
        #median-unbiased method for interpolating quantiles since distribution is unknown
        self.agg_results = self.results.groupby(['Unlabeled Percent']).\
            agg([np.mean, np.std,
                 lambda x: np.quantile(a=x, q=0.025, method = 'median_unbiased'),
                 lambda x: np.quantile(a=x, q=0.975, method = 'median_unbiased')
                 ])        
                
    def plot(self):
        """
        Plots the results of the experiment. Each model has its own plot with
        the percentage of unlabeled data on the x-axis and the f1 score
        (if fully supervised) or the f1 score ratio (if semi-supervised)
        on the y-axis.
        
        Parameters
        ---------
        None
        
        Returns
        -------
        None        
        """        
       
        n_plot_cols = 3
        lo_model_names = ['LR', 'RF', 'KNN',
                           'STC-T-LR','STC-T-RF', 'STC-T-KNN',
                           'STC-K-LR', 'STC-K-RF', 'STC-K-KNN'] 
        
        fig, ax = plt.subplots(3,n_plot_cols, figsize=(10, 10))
        
        i = j = 0            
        
        for model_name in lo_model_names:
            d = self.agg_results[model_name]
            upper = d['<lambda_1>'] 
            lower = d['<lambda_0>']
            sns.lineplot(x = d.index, y = d['mean'], ax = ax[i][j])

            ax[i][j].fill_between(d.index,lower, upper, alpha=0.5)
            ax[i][j].set_title(model_name)
            ax[i][j].set_xlabel('')
            ax[i][j].set_ylabel('')
            
            if model_name[0:3] == 'STC':            
                ax[i][j].axhline(1, color = 'green')
            
            if j < n_plot_cols - 1:
                j += 1
            else:
                i += 1
                j = 0
        
        
        fig.supylabel('1st row: F1 Score | 2nd, 3rd row: F1 Score Ratio')
        fig.supxlabel('Percentage of Unlabeled Data')

        plt.tight_layout()