import pandas as pd
import numpy as np
import logging
from sklearn import preprocessing


class EncodeLabel(object):
    def __init__(self, filename):
        self.cols_to_convert = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                                'shopperinteraction', 'cardverificationcodesupplied', 'accountcode', 'cvcresponsecode']

        self.filename = filename

    def _load_data(self):
        clean_df = pd.read_csv(self.filename)
        return clean_df

    def encode_one_hot(self):
        clean_df = self._load_data()
        one_hot_df = pd.get_dummies(clean_df, columns=self.cols_to_convert)

        return one_hot_df

    def encode_binary_label(self, target_col):
        lb = preprocessing.LabelBinarizer()
        lb.fit(target_col)

        transform_label = lb.transform(target_col)
        return transform_label

    def encode_label(self, columns):
        lb = preprocessing.LabelEncoder()
        transform_label = lb.fit_transform(columns)
        return transform_label
