import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import preprocessing
from sklearn.decomposition import PCA


class dataset():
    def __init__(self, csvfile='./data/cc_general.csv', clean='dropna') -> None:
        """
        csvfile : data file
        clean   : data clean method, dropna or fillzero
        """
        df = pd.read_csv(csvfile, sep=',')
        df.drop(columns=['CUST_ID'], inplace=True)
        if clean == 'dropna':
            df.dropna(inplace=True)
        elif clean == 'fillzero':
            df.fillna(0)
        else:
            df.dropna(inplace=True)
        df.info()
        self.df = df
    def data(self, pca_dim=None, standard='standard'):
        value = self.df.values
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            value = pca.fit_transform(value)
        if standard == 'maxmin':
            scaler = preprocessing.MinMaxScaler()
            scaler.fit(value)
            value = scaler.transform(value)
        elif standard == 'standard':
            scaler = preprocessing.StandardScaler()
            scaler.fit(value)
            value = scaler.transform(value)
        return value