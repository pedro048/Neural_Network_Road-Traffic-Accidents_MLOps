"""
Creators: Pedro Victor and Beatriz Soares
Date: 23 July 2022
Create API
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Select a Feature
class FeatureSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what this custom transformer need to do
    def transform(self, X, y=None):
        return X[self.feature_names]

# Handling categorical features
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    # Class constructor method that takes one boolean as its argument
    def __init__(self, new_features=True, colnames=None):
        self.new_features = new_features
        self.colnames = colnames

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self):
        return self.colnames.tolist()

    # Transformer method we wrote for this transformer
    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self.colnames)

        # Remove white space in categorical features
        df = df.apply(lambda row: row.str.strip())

        # customize feature?
        # How can I identify what needs to be modified? EDA!!!!
        if self.new_features:

            # minimize the cardinality of native_country feature
            # check cardinality using df.native_country.unique()
            df.loc[df['Educational_level'] == 'Junior high school','Educational_level'] = 'High school'
            df.loc[df['Educational_level'] == 'Writing & reading','Educational_level'] = 'Elementary school'
            df.loc[df['Educational_level'] == 'Illiterate','Educational_level'] = 'Illiterate'

            df.loc[df['Driving_experience'] == 'Below 1yr','Driving_experience'] = 'Below 1yr-2yr'
            df.loc[df['Driving_experience'] == '1-2yr','Driving_experience'] = 'Below 1yr-2yr'

            df.loc[df['Weather_conditions'] == 'Fog or mist','Weather_conditions'] = 'Fog or mist'

        # update column names
        self.colnames = df.columns

        return df    
    
