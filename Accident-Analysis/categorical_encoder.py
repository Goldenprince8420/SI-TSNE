import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
# from sklearn.preprocessing import OneHotEncoder


class Encoder:
    def __init__(self, data, cat_columns):
        self.frame = data
        self.cat_cols = cat_columns
        self.encoder = None
        self.encoded_frame = self.frame.copy()
        self.encoder_onehot = None
        self.onehot_encoded_data = None

    def category_encode(self):
        self.encoder = OrdinalEncoder(dtype = np.float64, unknown_value = np.nan, handle_unknown = "use_encoded_value")
        for column in self.cat_cols:
            self.encoded_frame[column] = self.encoder.fit_transform(self.encoded_frame[[column]])

    def get_encoded_data(self):
        self.category_encode()
        return self.encoded_frame

    def encode_onehot(self, encode_columns):
        # encoder = OneHotEncoder(max_categories = 30, min_frequency = 5)
        self.onehot_encoded_data = pd.get_dummies(self.frame, columns = encode_columns)

    def get_onehot_encoded_data(self, columns):
        self.encode_onehot(columns)
        return self.onehot_encoded_data
