import pandas as pd
from sklearn import preprocessing


class EncodeLabel(object):
    @staticmethod
    def encode_one_hot(clean_df, cols):
        """
        Helper function to perform One-Hot Encoding
        :param clean_df: 
        :param cols: 
        :return: 
        """
        # clean_df = self.load_data()
        one_hot_df = pd.get_dummies(clean_df, columns=cols)

        return one_hot_df

    @staticmethod
    def encode_binary_label(target_col):
        """
        Helper function for Binary Encoding
        :param target_col: 
        :return: 
        """
        lb = preprocessing.LabelBinarizer()
        lb.fit(target_col)

        transform_label = lb.transform(target_col)
        return transform_label

    @staticmethod
    def encode_label(columns):
        """
        Helper function for Enocding Categorical labels
        :param columns: 
        :return: 
        """
        lb = preprocessing.LabelEncoder()
        transform_label = lb.fit_transform(columns)
        # print lb.transform(['Settled', 'Chargeback'])
        return transform_label
