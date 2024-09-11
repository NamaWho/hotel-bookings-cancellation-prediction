from sklearn.base import BaseEstimator, TransformerMixin

class Wrapper(BaseEstimator, TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.is_test = False
        return self.transformer.fit(X, y)
    
    def transform(self, X):
        self.transformer.is_test = True 
        return self.transformer.transform(X)
    
    def fit_transform(self, X, y=None):
        self.transformer.is_test = False  
        return self.transformer.fit_transform(X, y)