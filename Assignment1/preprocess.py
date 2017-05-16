import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler

from label_encoder import EncodeLabel

logging.basicConfig(filename='logs/process.log', filemode='w+', level=logging.DEBUG)


class PreprocessData(object):
    def __init__(self):
        self.filename = 'data_for_student_case.csv'
        self.cols_to_convert = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                                'shopperinteraction', 'cardverificationcodesupplied', 'accountcode', 'cvcresponsecode',
                                'bin', 'mail_id', 'ip_id', 'card_id']

        self.cat_cols = ['issuercountrycode', 'txvariantcode', 'currencycode', 'shoppercountrycode',
                         'shopperinteraction', 'cardverificationcodesupplied', 'accountcode', 'cvcresponsecode',
                         'bin', 'mail_id', 'ip_id', 'card_id']

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

    def encode_cat_to_labels(self):
        """
        Performs one-hot encoding using Pandas' .get_dummies() method
        :return: one-hot encoded features
        """
        clean_df = self.remove_refused()
        e = EncodeLabel()
        clean_df_one_hot = e.encode_one_hot(clean_df=clean_df, cols=self.cols_to_convert)

        return clean_df_one_hot

    def encode_target_label(self, target_col):
        """
        Encodes the categorical target column into integer type
        :param target_col: Integer encoded target column array
        :return: 
        """
        e = EncodeLabel()
        target_labels = e.encode_label(target_col)

        return target_labels

    def encode_multicols(self):
        """
        
        :return: 
        """
        e = EncodeLabel()
        clean_df_non_ref = self.remove_refused()

        for cols in self.cat_cols:
            col_to_encode = clean_df_non_ref[cols]
            clean_df_non_ref[cols] = e.encode_label(col_to_encode)

        return clean_df_non_ref

    def create_dataframe(self, classifier_type='tree'):

        if classifier_type == 'tree':
            clean_df_non_ref = self.encode_multicols()
            clean_df_non_ref['simple_journal'].replace('Settled', 0, inplace=True)
            clean_df_non_ref['simple_journal'].replace('Chargeback', 1, inplace=True)
            target_labels = clean_df_non_ref['simple_journal']
            amount_col = clean_df_non_ref['amount']
            encoded_features = clean_df_non_ref.ix[:, self.cat_cols]
            encoded_features['amount'] = amount_col
            logging.debug("DataFrame Info:{} ".format(encoded_features.info()))

            return encoded_features, target_labels

        else:
            clean_df_one_hot = self.encode_cat_to_labels()
            clean_df_one_hot['simple_journal'].replace('Settled', 0, inplace=True)
            clean_df_one_hot['simple_journal'].replace('Chargeback', 1, inplace=True)
            target_labels = clean_df_one_hot['simple_journal']
            amount_col = clean_df_one_hot['amount']

            # s = StandardScaler()
            # amount_col_scaled = s.fit_transform(amount_col)

            encoded_features = clean_df_one_hot.ix[:, 7:]
            # target_labels = self.encode_target_label(target_col)
            # encoded_features['target'] = target_labels
            encoded_features['amount'] = amount_col

            logging.debug("DataFrame Info:{} ".format(encoded_features.info()))
            # logging.info("First 5 entries: {}".format(encoded_features.head()))

            return encoded_features, target_labels

# if __name__ == '__main__':
#     p = PreprocessData()
#     p.create_dataframe()
