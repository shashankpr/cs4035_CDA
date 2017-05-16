from preprocess import PreprocessData
p = PreprocessData()

class FeatureExtraction(object):
    def __init__(self):
        self.clean_df = p.remove_refused()

    def aggregate_by_email(self):
        chargeback_amt_email = self.clean_df[self.clean_df['simple_journal'] == 'Chargeback'].groupby('mail_id')[
            'amount']
        self.clean_df['mail_amt_count'] = chargeback_amt_email.transform('count')
        self.clean_df['mail_amt_sum'] = chargeback_amt_email.transform('sum')

    def aggregate_by_ip(self):
        chargeback_amt_ip = self.clean_df[self.clean_df['simple_journal'] == 'Chargeback'].groupby('ip_id')[
            'amount']
        self.clean_df['ip_amt_count'] = chargeback_amt_ip.transform('count')
        self.clean_df['ip_amt_sum'] = chargeback_amt_ip.transform('sum')

    def aggregate_by_card(self):
        chargeback_amt_card = self.clean_df[self.clean_df['simple_journal'] == 'Chargeback'].groupby('card_id')[
            'amount']
        self.clean_df['card_amt_count'] = chargeback_amt_card.transform('count')
        self.clean_df['card_amt_sum'] = chargeback_amt_card.transform('sum')

    def aggregate_by_cvc(self):
        chargeback_amt_cvc = \
        self.clean_df[self.clean_df['simple_journal'] == 'Chargeback'].groupby('cardverificationcodesupplied')[
            'amount']
        self.clean_df['cvc_amt_count'] = chargeback_amt_cvc.transform('count')
        self.clean_df['cvc_amt_sum'] = chargeback_amt_cvc.transform('sum')

    def get_new_features(self):
        return self.clean_df


if __name__ == '__main__':
    f = FeatureExtraction()
    f.aggregate_email()
