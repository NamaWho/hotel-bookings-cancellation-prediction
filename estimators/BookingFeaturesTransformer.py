from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta

class BookingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.third_quartile_adr_ = None     
        pass
    
    def fit(self, X, y=None):   
        X = X.copy()
        
        # Calculate the 75th percentile of ADR for each group using only the training data
        self.third_quartile_adr_ = X.groupby(['DistributionChannel', 'ReservedRoomType', 'ArrivalDateYear', 'ArrivalDateWeekNumber'])['ADR'].quantile(0.75).reset_index()
        self.third_quartile_adr_.rename(columns={'ADR': 'ThirdQuartileADR'}, inplace=True)
        
        return self
    
    def transform(self, X):        
        X = X.copy()

        # Merge the pre-calculated 75th percentile ADR values with the dataset
        X = X.merge(self.third_quartile_adr_, on=['DistributionChannel', 'ReservedRoomType', 'ArrivalDateYear', 'ArrivalDateWeekNumber'], how='left')

        # Calculate ADRThirdQuartileDeviation
        X['ADRThirdQuartileDeviation'] = X.apply(
            lambda row: row['ADR'] / row['ThirdQuartileADR'] if row['ThirdQuartileADR'] > 0 else 0,
            axis=1
        )

        X = X.drop(columns=['ADR', 
                            'ThirdQuartileADR',
                            'ArrivalDateYear',
                            'ArrivalDateMonth', 
                            'ArrivalDateWeekNumber',
                            'ArrivalDateDayOfMonth', 
                            'ReservedRoomType'])
        
        return X
