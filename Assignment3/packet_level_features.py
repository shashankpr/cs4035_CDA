import pandas as pd
import numpy as np
import logging
import preprocess

logging.basicConfig(level=logging.DEBUG)



def encode_categorical(df):
    df.drop(['StartTime', 'Dir', 'sTos', 'dTos', 'State'], axis=1, inplace=True)
    cols_to_enc = ['DstAddr', 'Proto']
    one_hot_df = pd.get_dummies(df, columns=cols_to_enc)

    logging.debug(one_hot_df.info())
    return one_hot_df

def create_new_df(one_hot_df):
    labels = one_hot_df['Label']
    one_hot_df.drop(['Label'], axis=1, inplace=True)
    features = one_hot_df

    logging.debug(features.shape)
    logging.debug(labels.shape)
    return features, labels

