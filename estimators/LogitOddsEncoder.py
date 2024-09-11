from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class LogitOddsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col='IsCanceled', min_frequency=0.02, smoothing=1e-6):
        self.target_col = target_col
        self.min_frequency = min_frequency
        self.smoothing = smoothing
        self.logit_odds_map = {}
        self.global_mean_logit_odds = None
    
    def fit(self, X, y=None):
        # Store global logit odds as a fallback for unseen categories
        global_mean = y.mean()
        self.global_mean_logit_odds = np.log(global_mean / (1 - global_mean))

        for col in X.columns:
            counts = X[col].value_counts(normalize=True)
            common_levels = counts[counts >= self.min_frequency].index
            group_data = X[[col]].copy()
            group_data[self.target_col] = y
            
            # Calculate logit odds with proper handling of 0 and 1 values
            logit_odds = group_data.groupby(col)[self.target_col].mean().apply(
                lambda x: np.log(np.clip(x, self.smoothing, 1 - self.smoothing) / 
                                 (1 - np.clip(x, self.smoothing, 1 - self.smoothing)))
            )
            self.logit_odds_map[col] = logit_odds.reindex(common_levels).fillna(self.global_mean_logit_odds)
        return self
    
    def transform(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            # Map the logit odds to the column, filling with the global mean logit odds if not found
            X_encoded[col] = X[col].map(self.logit_odds_map[col]).fillna(self.global_mean_logit_odds)
        return X_encoded