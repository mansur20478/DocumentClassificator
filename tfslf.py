import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


class TfslfVectorizer():
    '''TF-SLF vectorizer'''
    def __init__(self, **kwargs) -> None:
        self.counter = CountVectorizer(**kwargs)
    
    @staticmethod
    def _get_tf(X):
        '''Transform document-term count to frequencies.'''
        return X


    def fit(self, X, y):
        '''Fits X, y.
           X: list of texts.
           y: list of thier categories.'''
        X2 = self.counter.fit_transform(X)
        C = y.max() + 1 # category number
        R = 0
        for c in range(0, C):
            caterogy_matrix = X2[y == c]
            nc = caterogy_matrix.shape[0] # Num. documents            
            dfs = (caterogy_matrix != 0).sum(0)
            ndfs = dfs / nc
            R = R + ndfs
        self.slf_ = np.log(C / R)
        return self

    def transform(self, X, y=None):
        X2 = self.counter.transform(X)
        # transform count to frequencies
        X2 = X2.multiply(self.slf_)
        X2 = normalize(X2, norm='l2', axis=1)
        return X2
        

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)