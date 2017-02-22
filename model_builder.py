import mmh3
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def logistic_model(X_train, y_train):
    """
    use crossvalidation (CV) to report the best parameter 'C'
    parameter C: Inverse of regularization strength; must be a positive float. 
    Check LogisticRegression() in sklearn for more information
    """
    print('Train Regression Model')
    model = GridSearchCV(
            estimator=LogisticRegression(),
            param_grid={'C': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]},
            scoring='log_loss',
            cv=5
    )
    model.fit(X_train, y_train)
    return model
def preprocessing_data(df,numerical_list,string_list,int_list,timestamp_list):
    """
    fill numerical data with avg
    string_list na ->'UNK'
    int_list int->string -> na->'UNK'
    timestamp_list timestamp->hour[0-23]->string na->UNK

    notice: don't include is_install in your int_list!!!!
    check with my data_cleaning notebook

    """
    columns = {'is_install':int}
    df = set_column_types(df, columns)

    df = convert_to_string(df,int_list)
    #df = convert_to_unknown(df,int_list)
    df = convert_to_hour(df,timestamp_list)
    df = convert_to_unknown(df,timestamp_list+int_list+string_list)    
    df = fillna0(df, columns.keys())
    df = fillna_avg(df,numerical_list)
    
    return df

def convert_to_hour(df, columns):
    """
    convert NA to 'UNK'
    """
    if isinstance(df, pd.DataFrame):
        for col in columns:
            df[col]= pd.to_datetime(df[col])
            df[col]=df[col].dt.hour.astype(str)

    
    return df

def convert_to_unknown(df, columns):
    """
    convert NA to 'UNK'
    """
    if isinstance(df, pd.DataFrame):
        for col in columns:
            df[col].fillna("UNK", inplace=True)

    if isinstance(df, dict):
        for col in columns:
            df[col][pd.isnull(df[col])] = "UNK"
    return df
def convert_to_string(df, columns):
    """
    convert NA to 'UNK'
    """
    if isinstance(df, pd.DataFrame):
        for col in columns:
            df[col] = df[col].astype(str)

    return df

          
    
def fillna_avg(df, columns):
    """
    fill NA with mean
    """
    if isinstance(df, pd.DataFrame):
        for col in columns:
            df[col].fillna(df[col].mean(), inplace=True)
   
    return df

def fillna0(df, columns):
    """
    fill NA with 0
    """
    if isinstance(df, pd.DataFrame):
        for col in columns:
            df[col].fillna(0, inplace=True)

    if isinstance(df, dict):
        for col in columns:
            df[col][pd.isnull(df[col])] = 0    
    return df


def set_column_types(df, column_types_dict):
    if isinstance(column_types_dict, dict):
        for c, t in column_types_dict.items():
            df[c] = df[c].astype(t)
        return df 
    else:
        raise TypeError()


  
class FeatureCreator():
    """Augment DataFrame-like input with new features."""

    def transform(self, X, inplace=False):
        # TODO probably it's a good idea to restrict what fields from the
        #      DataFrame are used to avoid copying the whole thing
        if isinstance(X, pd.DataFrame):
            X = {k: v.values for k, v in X.iteritems()}
        if not inplace:
            X = {k: np.copy(v) for k, v in X.iteritems()}
        return X

    def fit(self, X, y=None):
        return self


# Hashing function
def _murmur_32s(key, seed):

    if isinstance(key, unicode):
        bkey = key.encode('utf-8')
    elif isinstance(key, bytes):
        bkey = key
    else:
        print key
        print type(key)
        raise ValueError("the key must be either unicode or str")
    return mmh3.hash(bkey, seed)

  
# Hash features of DataFrame X using the hashing function
def _transform(X, n_bits, categorical_features,
              continuous_features, interaction_features,
              store_fmap=False):
    n_samples = X.shape[0] \
        if isinstance(X, pd.DataFrame) \
        else len(X.values()[0])
    hash_mask = 2 ** n_bits - 1
    n_features = \
        len(categorical_features) + \
        len(continuous_features) + \
        len(interaction_features)
    n_hashed_features = n_samples * n_features
    # assert n_hashed_features > 0
    rows = np.empty(n_hashed_features, dtype=np.int32)
    cols = np.empty(n_hashed_features, dtype=np.int32)
    vals = np.zeros(n_hashed_features)
    hashed_feature_idx = 0
    f_map = {}

    for f in categorical_features:
        Xf = X[f]
        hash_seed = _murmur_32s(f, 0)
        for sample_idx in range(n_samples):
            hash_value = _murmur_32s(Xf[sample_idx], hash_seed)
            hash_sign = (hash_value >= 0) * 2 - 1

            if store_fmap:
                f_combined = ((f,), Xf[sample_idx])
                if f_combined not in f_map:
                    f_map[f_combined] = hash_value & hash_mask

            rows[hashed_feature_idx] = sample_idx
            cols[hashed_feature_idx] = hash_value & hash_mask
            vals[hashed_feature_idx] += hash_sign
            hashed_feature_idx += 1

    for f in continuous_features:
        Xf = X[f]
        hash_value = _murmur_32s(f, 0)
        hash_sign = (hash_value >= 0) * 2 - 1
        if store_fmap:
            f_combined = ((f,),)
            f_map[f_combined] = hash_value & hash_mask
        for sample_idx in range(n_samples):
            rows[hashed_feature_idx] = sample_idx
            cols[hashed_feature_idx] = hash_value & hash_mask
            vals[hashed_feature_idx] += hash_sign * Xf[sample_idx]
            hashed_feature_idx += 1

    for feature_names in interaction_features:
        hash_seed = 0
        for f in feature_names:
            hash_seed = _murmur_32s(f, hash_seed)

        for sample_idx in range(n_samples):
            hash_value = hash_seed
            interaction_value = 1

            value_cache = ()
            for f in feature_names:
                if f in continuous_features:
                    interaction_value *= X[f][sample_idx]
                    value_cache += (f,)
                else:
                    value_cache += (X[f][sample_idx],)
                    hash_value = _murmur_32s(
                        X[f][sample_idx], hash_value
                    )

            if store_fmap:
                f_combined = (feature_names, value_cache)
                if f_combined not in f_map:
                    f_map[f_combined] = hash_value & hash_mask

            hash_sign = (hash_value >= 0) * 2 - 1
            rows[hashed_feature_idx] = sample_idx
            cols[hashed_feature_idx] = hash_value & hash_mask
            vals[hashed_feature_idx] += hash_sign * interaction_value
            hashed_feature_idx += 1

    n_dim_hashed_features = hash_mask + 1

    # reverse k and v, if v is duplicated, append k to v
    f_map_rev = {}
    for k, v in f_map.items():
        if v not in f_map_rev:
            f_map_rev[v] = [k]
        else:
            f_map_rev[v] = f_map_rev[v] + [k]

    return sparse.coo_matrix(
        (vals, (rows, cols)),
        (n_samples, n_dim_hashed_features)
    ).tocsr(), f_map_rev

# Wrapper class for hashing function
class FeatureHasher(BaseEstimator, TransformerMixin):
    def __init__(self,
                 n_bits=22,
                 categorical_features=None,
                 continuous_features=None,
                 interaction_features=None,
                 store_fmap=False):
        if n_bits < 1 or n_bits > 31:
            raise ValueError("number of bits must be in interval [1, 31]")

        self.n_bits_ = n_bits
        self.categorical_features_ = set(categorical_features or [])
        self.continuous_features_ = set(continuous_features or [])
        self.interaction_features_ = set(interaction_features or [])
        self.store_fmap = store_fmap

        n_features = len(self.categorical_features_) + \
                     len(self.continuous_features_) + \
                     len(self.interaction_features_)
        if n_features == 0:
            raise ValueError("at least one features needs to be specified")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _transform(
            X, self.n_bits_, self.categorical_features_,
            self.continuous_features_, self.interaction_features_,
            self.store_fmap
        )










