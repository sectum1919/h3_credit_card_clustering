import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn import preprocessing
from sklearn.decomposition import PCA


class dataset():
    def __init__(self, csvfile='./data/cc_general.csv', clean=True) -> None:
        """
        csvfile : data file
        clean   : True or False
        """
        df = pd.read_csv(csvfile, sep=',')
        df.info()
        df.drop(columns=['CUST_ID'], inplace=True)
        if clean:
            df.loc[df["PAYMENTS"]<0.00001, 'MINIMUM_PAYMENTS'] = 0.0
            df.loc[df["PURCHASES_TRX"]==0, 'MINIMUM_PAYMENTS'] = 0.0
            df.dropna(inplace=True)
        df.info()
        self.df = df
    def data(self, pca_dim=None, standard='robust'):
        value = self.df.values
        if standard is not None:
            if standard == 'minmax':
                scaler = preprocessing.MinMaxScaler()
            if standard == 'maxabs':
                scaler = preprocessing.MaxAbsScaler()
            elif standard == 'standard':
                scaler = preprocessing.StandardScaler()
            elif standard == 'robust':
                scaler = preprocessing.RobustScaler()
            scaler.fit(value)
            value = scaler.transform(value)
        if pca_dim is not None:
            pca = PCA(n_components=pca_dim)
            value = pca.fit_transform(value)
        return value

if __name__ == '__main__':
    ccdata = dataset('./data/cc_general.csv', clean = False)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 313)
    print(ccdata.df['CREDIT_LIMIT'])
    print("MIN   ", ccdata.df['CREDIT_LIMIT'].min())
    print("MAX   ", ccdata.df['CREDIT_LIMIT'].max())
    print("MEAN  ", ccdata.df['CREDIT_LIMIT'].mean())
    print("MEDIAN", ccdata.df['CREDIT_LIMIT'].median())
    print("VAR   ", ccdata.df['CREDIT_LIMIT'].var())
    print(ccdata.df['MINIMUM_PAYMENTS'])
    print("MIN   ", ccdata.df['MINIMUM_PAYMENTS'].min())
    print("MAX   ", ccdata.df['MINIMUM_PAYMENTS'].max())
    print("MEAN  ", ccdata.df['MINIMUM_PAYMENTS'].mean())
    print("MEDIAN", ccdata.df['MINIMUM_PAYMENTS'].median())
    print("VAR   ", ccdata.df['MINIMUM_PAYMENTS'].var())

    print(ccdata.df[:][np.isnan(ccdata.df['CREDIT_LIMIT'])])
    
    ndf = ccdata.df[:][np.isnan(ccdata.df['MINIMUM_PAYMENTS'])]
    ndf.info()
    print("MIN   ", ndf["PAYMENTS"].min())
    print("MAX   ", ndf["PAYMENTS"].max())
    print("MEAN  ", ndf["PAYMENTS"].mean())
    print("MEDIAN", ndf["PAYMENTS"].median())
    print("VAR   ", ndf["PAYMENTS"].var())
    ndf["MINIMUM_PAYMENTS"][ndf["PAYMENTS"]<0.00001] = 0.0
    ndf["MINIMUM_PAYMENTS"][ndf["PURCHASES_TRX"]==0] = 0.0
    # print(ndf["PURCHASES_TRX"])
    ndf = ndf[:][np.isnan(ndf['MINIMUM_PAYMENTS'])]
    ndf.info()
    print(ndf["PURCHASES_TRX"].values)
    print(ndf["PURCHASES"].values)