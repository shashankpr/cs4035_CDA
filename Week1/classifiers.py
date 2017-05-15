import pandas as pd
import numpy as np
import logging
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import linear_model

from sampling import DataSampling
from preprocess import PreprocessData
from metrics import ClassificationMetric


class Classifier(object):
    def __init__(self):
        self.c_metric = ClassificationMetric()

    @staticmethod
    def _get_features_labels():
        p = PreprocessData()
        features, labels = p.create_dataframe()

        return features, labels

    @staticmethod
    def train_test_split(features, labels):
        X_train, X_test, y_train, y_test = train_test_split(features, labels)

        logging.debug("Training label size : {}".format(Counter(y_train)))
        logging.debug("Test label size: {}".format(Counter(y_test)))
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _get_sampled_features_labels(features, labels):
        s = DataSampling()
        features_res, labels_res = s.simulate_smote(features, labels)

        logging.debug("Resampled Train label size: {}".format(Counter(labels_res)))
        return features_res, labels_res

    def _get_data_for_classifier(self):
        features, labels = self._get_features_labels()
        X_train, X_test, y_train, y_test = self.train_test_split(features, labels)
        X_train_resampled, y_train_resampled = self._get_sampled_features_labels(X_train, y_train)

        return X_train_resampled, y_train_resampled, X_test, y_test

    def run_logistic_reg(self):
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        logreg = linear_model.LogisticRegression(C=1e5)
        logreg.fit(X_train_resampled, y_train_resampled)
        predicted = logreg.predict(X_test)

        logging.debug("Predicted Label count: {}".format(Counter(predicted)))

        precision_score = self.c_metric.compute_precision_score(y_test, predicted)
        recall_score = self.c_metric.compute_recall_score(y_test, predicted)
        c_report = self.c_metric.classification_report(y_test, predicted)

        return precision_score, recall_score, c_report

    def run_logistic_reg_cv(self):
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        logregcv = linear_model.LogisticRegressionCV(solver='sag', n_jobs=-1)
        logregcv.fit(X_train_resampled, y_train_resampled)
        predicted = logregcv.predict(X_test)

        logging.debug("Predicted Label count: {}".format(Counter(predicted)))

        precision_score = self.c_metric.compute_precision_score(y_test, predicted)
        recall_score = self.c_metric.compute_recall_score(y_test, predicted)
        c_report = self.c_metric.classification_report(y_test, predicted)

        return precision_score, recall_score, c_report

    def run_svc(self):
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        clf = svm.SVC()
        clf.fit(X_train_resampled, y_train_resampled)

        predicted = clf.predict(X_test)
        precision_score = self.c_metric.compute_precision_score(y_test, predicted)
        recall_score = self.c_metric.compute_recall_score(y_test, predicted)

        return precision_score, recall_score

    def run_decision_trees(self):
        pass


if __name__ == '__main__':
    c = Classifier()
    c.run_logistic_reg_cv()
