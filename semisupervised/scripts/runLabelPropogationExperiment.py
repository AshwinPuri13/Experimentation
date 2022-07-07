import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelPropagation

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

from random import sample

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

class runExperiment:
    """
    
    This class trains and evaluates several different supervised classification
    models on a given dataset where each of 10%, 20%, ..., 90% of the labels are
    intentionally removed and when those labels are relabeled through label propogation.
    
    Parameters:
    -----------
    n_splits: int
        Number of stratified samples. Each model, at each percentage of missing
        labels, is trained individually on each sample.
    
    test_size: float
        Size (percentage of total) of the test set. Should be in (0,1).
    
    f1_average_param: {'micro', 'macro', 'samples', 'weighted', 'binary'} or None
        The 'average' parameter of sklearn.metrics.f1_score.
        
    
    lo_unlabeled_pct: list(float)
        The percentages of missing data to train each model on. Each element 
        should be in (0,1).
        
        
    gamma: float
        The gamma parameter for label propogation. Only relevant when the kernel is rbf.
    
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
    
    def __init__(self, n_splits, test_size, f1_average_param, max_iter = 1000,
                 lo_unlabeled_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                 gamma = 20):
        
        #gamma = 20 is the sklearn LabelPropogation defaul
        
        self.n_splits = n_splits
        self.test_size = test_size
        self.lo_unlabeled_pct = lo_unlabeled_pct
        self.f1_average_param = f1_average_param
        self.gamma = gamma
        self.max_iter = max_iter
        
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
           
        self.results = pd.DataFrame()
        
        #we dont include LP into the pipeline because it has the possibility of relabeling labeled data
        #so we use LP outside
        do_models = {
            'LR': LogisticRegression(max_iter = 500),
            
            'RF': RandomForestClassifier(warm_start = False),
            
            'KNN': KNeighborsClassifier(),
            
            'SVM': SVC(),
            
            'LP-R-LR': LogisticRegression(max_iter = 500),
            
            'LP-R-RF': RandomForestClassifier(warm_start = False),
            
            'LP-R-KNN': KNeighborsClassifier(),
            
            'LP-R-SVM': SVC(),
            
            'LP-K-LR': LogisticRegression(max_iter = 500),
            
            'LP-K-RF': RandomForestClassifier(warm_start = False),
            
            'LP-K-KNN': KNeighborsClassifier(),
            
            'LP-K-SVM': SVC()
            }
        
        # modified version of the rbf kernel to prevent overflow
        # https://stackoverflow.com/questions/52057836/labelpropagation-how-to-avoid-division-by-zero
        def rbf_kernel_safe(X, Y = None, gamma = None): 
            
          from sklearn.metrics.pairwise import check_pairwise_arrays
          from sklearn.metrics.pairwise import euclidean_distances

          X, Y = check_pairwise_arrays(X, Y) 
          if gamma is None: 
              gamma = 1.0 / X.shape[1] 
    
          K = euclidean_distances(X, Y, squared=True) 
          K *= -gamma 
          K -= K.max()
          np.exp(K, K)    # exponentiate K in-place 
          return K 

        #using shuffle split since test size needs to be sufficient for f1 score to be meaningful
        kf = StratifiedShuffleSplit(n_splits = self.n_splits,
                                    test_size = self.test_size)
        
        for train_index, test_index in kf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for unlabeled_pct in self.lo_unlabeled_pct:
                y_train_unlab = y_train.copy()   
                unlabel_set = sample(list(y_train_unlab.index),
                                     k = int(unlabeled_pct * len(y_train_unlab)))
                y_train_unlab[unlabel_set] = -1

                do_scores = {'Unlabeled Percent': [unlabeled_pct]}      
                LPK = LabelPropagation(kernel = 'knn', max_iter = self.max_iter)
                LPR =  LabelPropagation(kernel = lambda X, Y: rbf_kernel_safe(X, Y, self.gamma),
                                        max_iter = self.max_iter)
                
                LPK.fit(X_train, y_train_unlab)
                LPR.fit(X_train, y_train_unlab)

                y_train_relab_k = y_train_unlab.copy()
                y_train_relab_k[unlabel_set] = pd.Series(LPK.transduction_,
                                                       index = y_train_unlab.index)[unlabel_set]
                y_train_relab_r = y_train_unlab.copy()
                y_train_relab_r[unlabel_set] = pd.Series(LPR.transduction_,
                                                       index = y_train_unlab.index)[unlabel_set]                                                           
                
                for model_name in do_models.keys():
                    if model_name[0:2] == 'LP':  
                        
                        if model_name[3] == 'K':
                            do_models[model_name].\
                                fit(X_train, y_train_relab_k)
                        else:
                            do_models[model_name].\
                                fit(X_train, y_train_relab_r)                      
            
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
            if model_name[0:2] == 'LP':
                self.results[model_name] /= self.results[model_name[5:]]                
                
                
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
        
        fig, ax = plt.subplots(3,4, figsize=(10, 10))
        
        i = j = 0            
        
        for model_name in list(self.results)[1:]:
            d = self.agg_results[model_name]
            upper = d['<lambda_1>'] 
            lower = d['<lambda_0>']
            sns.lineplot(x = d.index, y = d['mean'], ax = ax[i][j])

            ax[i][j].fill_between(d.index,lower, upper, alpha=0.5)
            ax[i][j].set_title(model_name)
            ax[i][j].set_xlabel('')
            ax[i][j].set_ylabel('')
            
            if model_name[0:2] == 'LP':            
                ax[i][j].axhline(1, color = 'green')
            
            if j < 3:
                j += 1
            else:
                i += 1
                j = 0
        
        fig.supylabel('1st row: F1 Score | 2nd, 3rd row: F1 Score Ratio')
        fig.supxlabel('Percentage of Unlabeled Data')

        plt.tight_layout()