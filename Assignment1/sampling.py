import logging
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

from preprocess import PreprocessData


class DataSampling(object):
    def __init__(self):
        self.filename = 'clean_non_refused.csv'

    def _load_data(self):
        clean_df = pd.read_csv(self.filename)
        return clean_df

    @staticmethod
    def simulate_smote(features, labels):
        # logging.info('Original dataset shape {}'.format(Counter(labels)))

        sm = SMOTE(random_state=0)
        features_res, labels_res = sm.fit_sample(features, labels)
        # logging.info('Resampled dataset shape {}'.format(Counter(labels_res)))

        return features_res, labels_res
