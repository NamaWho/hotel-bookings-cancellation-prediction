from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta

class BookingFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, is_test=False, drop_columns=[]):
        self.is_test = is_test
        self.drop_columns = drop_columns
        pass
    
    def fit(self, X, y=None):        
        return self
    
    def transform(self, X):        
        X = X.copy()

        # Create 'LiveTime' feature
        # X['LiveTime'] = X.apply(lambda row: self.calculate_livetime(X, row), axis=1)
        X['LiveTime'] = X['LeadTime']

        # Calculate the 75th percentile of ADR for each group
        X['ThirdQuartileADR'] = X.groupby(['DistributionChannel', 'ReservedRoomType', 'ArrivalDateYear'])['ADR'].transform(lambda x: x.quantile(0.75))

        # Calculate ADRThirdQuartileDeviation
        X['ADRThirdQuartileDeviation'] = X.apply(
            lambda row: row['ADR'] / row['ThirdQuartileADR'] if row['ThirdQuartileADR'] > 0 else 0,
            axis=1
        )

        X = X.drop(columns=['LeadTime',
                            'ADR', 
                            'ThirdQuartileADR',
                            'ArrivalDateYear',
                            'ArrivalDateMonth', 
                            'ArrivalDateDayOfMonth', 
                            'ReservationStatus', 
                            'ReservationStatusDate',
                            'ReservedRoomType'] + self.drop_columns)
        
        return X

    def calculate_livetime(self, X, row):
        arrival_date = datetime(row['ArrivalDateYear'], row['ArrivalDateMonth'], row['ArrivalDateDayOfMonth'])
        booking_date = arrival_date - timedelta(days=row['LeadTime'])

        if self.is_test:
            # 'C' booking type (current bookings)
            processing_date = arrival_date - timedelta(weeks=2)
            if (processing_date - booking_date).days < 0:
                processing_date = arrival_date - timedelta(weeks=1)
                if (processing_date - booking_date).days < 0:
                    return 0
                else:
                    return (processing_date - booking_date).days
            return (processing_date - booking_date).days
        else:
            if row['ReservationStatus'] == 'Check-Out':  # "A" type (effective bookings)
                return row['LeadTime']
            else:  # "B" type (canceled bookings or no show)
                reservation_status_date = datetime.strptime(row['ReservationStatusDate'], "%Y-%m-%d")
                # print('Canceled booking:', (reservation_status_date-booking_date).days)
                return (reservation_status_date - booking_date).days
