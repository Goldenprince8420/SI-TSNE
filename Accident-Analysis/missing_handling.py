import numpy as np
import pandas as pd
import sklearn as skl
import scipy
from read_data import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer, MissingIndicator


class MissingHandler:
    def __init__(self, data):
        self.frame = data
        self.strategy = None
        self.criterion = None
        self.imputed_data = None

    def get_missing_info(self):
        missing_values_count = self.frame.isnull().sum()
        # how many total missing values do we have?
        total_cells = np.product(self.frame.shape)
        total_missing = missing_values_count.sum()
        missing_perc = (total_missing / total_cells) * 100
        print("Total Missing Values in the Data: {}".format(total_missing))
        print("Missing values per column: {}".format(missing_values_count))
        print("Percentage of missing values: {}".format(missing_perc))
        return

    def simpleImputer(self, strategy='mean'):
        # create the mean imputer
        mi_mean_data = SimpleImputer(
            strategy=strategy
        )
        # print the mean imputer to console
        print(mi_mean_data)
        # perform mean imputation procedure
        imp_mean_data = mi_mean_data.fit_transform(self.frame)
        print("Imputation Successful!")
        return imp_mean_data

    def impute_missing_values(self, type_imputer, strategy):
        if type_imputer == "simple":
            self.imputed_data = self.simpleImputer(strategy = strategy)
        elif type_imputer == "iterative":
            self.imputed_data = self.iterative_imputer()
        else:
            print("Invalid Imputer Type!!")
        return self.imputed_data







