import pandas as pd
import numpy as np
import logging

logging.basicConfig(filename='logs/process.log', filemode='w+', level=logging.DEBUG)

class PreprocessData(object):
    def __init__(self):
        self.filename = 'data_for_student_case.csv'

    def _load_dataset(self):
        fraud_df = pd.read_csv(self.filename)
        return fraud_df

    def _remove_missing_values(self):
        fraud_df = self._load_dataset()
        clean_df = fraud_df.dropna()

        return clean_df

    def _combine_cvccode(self):
        clean_df = self._remove_missing_values()

        if 4 in clean_df['cvcresponsecode']:
            clean_df['cvcresponsecode'].replace(4, 3, inplace=True)
        if 5 in clean_df['cvcresponsecode']:
            clean_df['cvcresponsecode'].replace(5, 3, inplace=True)
        if 6 in clean_df['cvcresponsecode']:
            clean_df['cvcresponsecode'].replace(6, 3, inplace=True)

        return clean_df

    def _drop_columns(self):
        clean_df = self._combine_cvccode()
        clean_df.drop(['txid', 'bookingdate'], axis=1, inplace=True)

        return clean_df

    def remove_refused(self, save_file=False):
        clean_df = self._drop_columns()

        clean_df_non_ref = clean_df[clean_df['simple_journal'] != 'Refused']

        if save_file:
            pd.DataFrame(clean_df_non_ref).to_csv('clean_non_refused.csv', index=False)

        total_count = clean_df_non_ref['simple_journal'].count()

        logging.info("The count of labels in the processed dataset:")
        logging.debug(clean_df_non_ref['simple_journal'].value_counts())

        logging.info("Percentage of labels:")
        logging.debug(clean_df_non_ref['simple_journal'].value_counts() / total_count)

        return clean_df_non_ref

if __name__ == '__main__':
    p = PreprocessData()
    p.remove_refused(save_file=True)
