import pandas as pd
import numpy as np
import logging
from sklearn import preprocessing


class EncodeLabel(object):
    @staticmethod
    def encode_one_hot(clean_df, cols):
        # clean_df = self.load_data()
        one_hot_df = pd.get_dummies(clean_df, columns=cols)

        return one_hot_df

    @staticmethod
    def encode_binary_label(target_col):
        lb = preprocessing.LabelBinarizer()
        lb.fit(target_col)

        transform_label = lb.transform(target_col)
        return transform_label

    @staticmethod
    def encode_label(columns):
        lb = preprocessing.LabelEncoder()
        transform_label = lb.fit_transform(columns)
        return transform_label
