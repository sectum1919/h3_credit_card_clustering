import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


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
    def data(self):
        return self.df.values