import pandas as pd
import numpy as np
import matplotlib as plt
from collections import Counter


def load_csv():
    normal_df = pd.read_csv("dataset/Sensor_data_Normal.csv", parse_dates=True, index_col='Timestamp')
    attack_df = pd.read_csv("dataset/Sensor_data_NA.csv", parse_dates=True, index_col='Timestamp')

    return normal_df, attack_df


def create_train_test_set():
    normal_df, attack_df = load_csv()

    # Collect data till the first occurence of "Attack" (or the last occurence of "Normal")
    first_occurence = attack_df['Normal/Attack'].ne('Normal').idxmax()
    normal_vals_from_attack = attack_df[attack_df.index < first_occurence]
    training_data = normal_df.append(normal_vals_from_attack)
    training_timestamp = training_data.index

    # Store training labels and remove them from the data
    training_labels = training_data['Normal/Attack']
    training_data.drop(['Normal/Attack'], axis=1, inplace=True)

    # Reset index of training data and drop the Timestamp column
    training_data.reset_index(drop=True, inplace=True)

    # Add the rest of the data to test set
    test_data = attack_df[attack_df.index >= first_occurence]

    # Get test timestamp and drop them
    test_timestamp = test_data.index

    # Get test labels
    test_labels = test_data['Normal/Attack']

    test_data.drop(['Normal/Attack'], axis=1, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    print Counter(test_labels)

    assert training_data.shape == (498554, 51)
    assert test_data.shape == (448165, 51)

    return training_data, training_labels, training_timestamp, test_data, test_labels, test_timestamp


def to_array(train_set, test_set):
    train_set = np.asarray(train_set)
    test_set = np.asarray(test_set)


if __name__ == '__main__':
    create_train_test_set()
