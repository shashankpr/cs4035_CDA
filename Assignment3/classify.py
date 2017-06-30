import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

import preprocess, packet_level_features, feature_extraction
import logging

logging.basicConfig(level=logging.DEBUG)


def get_train_test(features, labels):
    src_ips = features['Src_IPs']
    features.drop(['Src_IPs'], axis=1, inplace=True)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

    logging.debug("Training label size = {}".format(Counter(y_train)))
    logging.debug("Test label size = {}".format(Counter(y_test)))

    return x_train, x_test, y_train, y_test


def run_cross_validation(x_train, y_train):
    kcv = KFold(n_splits=10)
    clf = RandomForestClassifier(random_state=42)
    cross_scores = cross_val_score(clf, x_train, y_train, cv=kcv, scoring='accuracy')

    logging.info("Cross validation score = {}".format(cross_scores))

    return cross_scores


def run_random_forest(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(x_train, y_train)
    predicted = clf.predict(x_test)
    pred_score = clf.score(x_test, y_test)

    logging.info("Prediction = {}".format(predicted))
    logging.info("Score = {}".format(pred_score))

    return predicted, pred_score


def full_metric_report(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred)
    recal_score = recall_score(y_test, y_pred)

    print "Precision"
    print prec_score

    print "Recall"
    print recal_score

    conf_report = pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print "Confusion matrix"
    print conf_report

    # target_names = ['class 0, class 1']
    # print "Classification Report"
    # print classification_report(y_test, y_pred, target_names)


if __name__ == '__main__':
    all_files = preprocess.load_data()
    training_df = preprocess.store_to_dataframe(all_files)
    processed_df = preprocess.preprocess_labels(training_df)
    # one_hot_df = packet_level_features.encode_categorical(processed_df)

    features, labels = feature_extraction.build_training_set(processed_df)

    x_train, x_test, y_train, y_test = get_train_test(features, labels)
    predicted, pred_scores = run_random_forest(x_train, x_test, y_train, y_test)

    # For cross-validation
    # cross_score = run_cross_validation(x_train, y_train)

    # For packet-level flow
    # one_hot_df = packet_level_features.encode_categorical(processed_df)
    # features, labels = feature_extraction.build_training_set(one_hot_df)
    # x_train, x_test, y_train, y_test = get_train_test(features, labels)
    # predicted, pred_scores = run_random_forest(x_train, x_test, y_train, y_test)

    full_metric_report(y_test, predicted)