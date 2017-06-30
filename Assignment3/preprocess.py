import pandas as pd
import numpy as np
from collections import Counter
import os
import glob
import re
import logging

# logging.basicConfig(level=logging.DEBUG)


def load_data():
    cwd = os.getcwd()
    filepaths = glob.glob(os.path.join(cwd + '/data/', "*.txt"))
    logging.debug(filepaths)

    return filepaths


def store_to_dataframe(filepaths):
    training_df = pd.concat(pd.read_csv(f) for f in filepaths)
    logging.info("Dataframe Info : {}".format(training_df.info()))

    return training_df


def preprocess_labels(df):
    # Replace all Background-related flow labels to 2
    df.loc[df['Label'].str.contains('Background', case=False, na=False), 'Label'] = 2

    # Replace Botnet-related flow labels to "botnet"
    df.loc[df['Label'].str.contains('Botnet', case=False, na=False), 'Label'] = 1

    # Replace Normal-related flow labels to 0
    df.loc[df['Label'].str.contains('Normal', case=False, na=False), 'Label'] = 0

    logging.debug("Number of class labels : {}".format(Counter(df['Label'])))

    # Remove the rows containg "background" as labels
    df = df[df.Label != 2]

    logging.debug("Number of new class labels : {}".format(Counter(df['Label'])))

    return df


# if __name__ == '__main__':
#     all_files = load_data()
#     training_df = store_to_dataframe(all_files)
#     processed_df = preprocess_labels(training_df)
