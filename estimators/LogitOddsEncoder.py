from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class LogitOddsEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, min_frequency=0.02, smoothing=1e-6):
        self.columns = columns
        self.min_frequency = min_frequency
        self.smoothing = smoothing
        self.logit_odds_map = {}
        self.prevalence_map = {}
        self.global_mean_logit_odds = None

    def get_feature_names_out(self, input_features=None):
        feature_names = []
        for col in self.logit_odds_map.keys():
            feature_names.append(f'{col}_logit_odds')
            feature_names.append(f'{col}_prevalence')
        return np.array(feature_names)
    
    def fit(self, X, y=None):
        # Ensure X is a DataFrame
        X = pd.DataFrame(X, columns=self.columns)
        y = pd.Series(y)

        # Store global logit odds as a fallback for unseen categories
        global_mean = y.mean()
        self.global_mean_logit_odds = np.log(global_mean / (1 - global_mean))

        for col in self.columns:
            counts = X[col].value_counts(normalize=True)
            # Identify common levels
            prevalence_flag = counts >= self.min_frequency
            self.prevalence_map[col] = prevalence_flag.astype(int)
            # Calculate mean target for each category level
            df = pd.DataFrame({col: X[col], 'target': y})
            mean_target = df.groupby(col)['target'].mean()
            # Calculate logit odds with smoothing
            logit_odds = mean_target.apply(
                lambda x: np.log(np.clip(x, self.smoothing, 1 - self.smoothing) /
                                 (1 - np.clip(x, self.smoothing, 1 - self.smoothing)))
            )
            self.logit_odds_map[col] = logit_odds
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns)
        X_encoded = pd.DataFrame(index=X.index)

        for col in self.columns:
            # Map logit odds, fill missing with global mean logit odds
            X_encoded[col + '_logit_odds'] = X[col].map(self.logit_odds_map[col]).fillna(self.global_mean_logit_odds)
            # Map prevalence flag, fill missing with 0 (rare)
            prevalence_series = self.prevalence_map[col]
            X_encoded[col + '_prevalence'] = X[col].map(prevalence_series).fillna(0).astype(int)
        return X_encoded