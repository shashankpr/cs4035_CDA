import pandas as pd
import numpy as np
import logging
from collections import Counter

from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

from sampling import DataSampling
from preprocess import PreprocessData
from metrics import ClassificationMetric


class Classifier(object):
    def __init__(self):
        self.c_metric = ClassificationMetric()

    @staticmethod
    def _get_features_labels():
        p = PreprocessData()
        features, labels = p.create_dataframe(classifier_type='nn')

        return features, labels

    @staticmethod
    def train_test_split(features, labels):
        """
        Split features and labels into Training and Test.
        :param features: 
        :param labels: 
        :return: 
        """
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

        logging.debug("Training label size : {}".format(Counter(y_train)))
        logging.debug("Test label size: {}".format(Counter(y_test)))
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _get_sampled_features_labels(features, labels):
        """
        Performs SMOTE on the training features.
        :param features: 
        :param labels: 
        :return: 
        """
        s = DataSampling()
        features_res, labels_res = s.simulate_smote(features, labels)

        logging.debug("Resampled Train label size: {}".format(Counter(labels_res)))
        return features_res, labels_res

    def _get_data_for_classifier(self):
        features, labels = self._get_features_labels()
        X_train, X_test, y_train, y_test = self.train_test_split(features, labels)
        X_train_resampled, y_train_resampled = self._get_sampled_features_labels(X_train, y_train)

        return X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train

    def get_data_for_classifier(self):
        features, labels = self._get_features_labels()
        X_train, X_test, y_train, y_test = self.train_test_split(features, labels)
        X_train_resampled, y_train_resampled = self._get_sampled_features_labels(X_train, y_train)

        return X_train_resampled, y_train_resampled, X_test, y_test

    def pca_decompose(self, features):
        """
        Initializes PCA transformation instance
        :param features: 
        :return: 
        """
        pca_init = PCA(n_components=3)
        pca_transform = pca_init.fit_transform(features)

        return pca_transform

    def _run_cross_validation(self):

        kcv = KFold(n_splits=10)
        time_cv = TimeSeriesSplit(n_splits=10)

        return kcv, time_cv

    def run_logistic_reg(self):
        """
        Run Logistic regression
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        logreg = linear_model.LogisticRegression(C=1e5)
        logreg.fit(X_train_resampled, y_train_resampled)
        predicted = logreg.predict(X_test)

        predicted_prob = logreg.predict_proba(X_test)
        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])

        logging.debug("Predicted Label count: {}".format(Counter(predicted)))

        return predicted

    def run_logistic_reg_cv(self):
        """
        Run Logistic Regression Cross validation
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        logregcv = linear_model.LogisticRegressionCV(solver='sag', n_jobs=-1)
        logregcv.fit(X_train_resampled, y_train_resampled)
        predicted = logregcv.predict(X_test)

        logging.debug("Predicted Label count: {}".format(Counter(predicted)))

        return predicted

    def run_svc(self):
        """
        RUN linear SVM. It is not suitable for our dataset due to large number of datapoints.
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        clf = svm.SVC(kernel='linear', probability=True)
        clf.fit(X_train_resampled, y_train_resampled)

        predicted = clf.predict(X_test)
        predicted_prob = clf.predict_proba(X_test)

        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])
        return predicted

    def run_decision_trees(self):
        """
        Run Decision tree classifier
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train = self._get_data_for_classifier()

        tree = DecisionTreeClassifier(min_samples_split=50, random_state=0)
        tree.fit(X_train_resampled, y_train_resampled)

        predicted = tree.predict(X_test)
        predicted_prob = tree.predict_proba(X_test)
        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])
        return predicted

    def run_random_forest(self):
        """
        Run Random Forest Classifier
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test = self._get_data_for_classifier()

        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train_resampled, y_train_resampled)
        predicted = clf.predict(X_test)

        predicted_prob = clf.predict_proba(X_test)
        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])

        return predicted

    def run_neural_network(self):
        """
        Run a simple Multilayer Perceptron with 15 hidden layer nodes. 
        PCA is used here to reduce the dimensionality
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train = self._get_data_for_classifier()

        X_pca_transf_train = self.pca_decompose(X_train_resampled)
        X_pca_transf_test = self.pca_decompose(X_test)
        clf = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,), random_state=0)
        clf.fit(X_pca_transf_train, y_train_resampled)

        predicted = clf.predict(X_pca_transf_test)
        predicted_prob = clf.predict_proba(X_pca_transf_test)

        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])

        return predicted

    def run_adaBoost(self):
        """
        Run AdaBoost Classifier
        :return: 
        """

        X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train = self._get_data_for_classifier()

        nn = MLPClassifier(solver='lbfgs', alpha=0.01, hidden_layer_sizes=(15,), random_state=0)
        clf1 = DecisionTreeClassifier(min_samples_split=20)
        clf = AdaBoostClassifier(n_estimators=100, base_estimator=clf1)
        clf.fit(X_train_resampled, y_train_resampled)

        predicted = clf.predict(X_test)
        predicted_prob = clf.predict_proba(X_test)
        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])

        return predicted

    def run_gradBoost(self):
        """
        Run GradientBoosting
        :return: 
        """

        X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train = self._get_data_for_classifier()

        clf = GradientBoostingClassifier(min_samples_split=20)
        clf.fit(X_train_resampled, y_train_resampled)

        predicted = clf.predict(X_test)
        predicted_prob = clf.predict_proba(X_test)
        self.c_metric.get_full_report(y_test, predicted, y_test, predicted_prob[:, 1])

    def run_majority_voting(self):
        """
        Run Voting Classifier
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train = self._get_data_for_classifier()

        clf1 = DecisionTreeClassifier(min_samples_split=50)
        clf2 = RandomForestClassifier()
        clf3 = MLPClassifier(hidden_layer_sizes=(15,), alpha=0.01)

        clfV = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2)])

        clfV.fit(X_train_resampled, y_train_resampled)
        predicted = clfV.predict(X_test)
        predicted_prob = clfV.predict_proba

        self.c_metric.get_full_report(y_test, predicted, roc_reqd=False)
        return clfV

    def exec_cross_validation(self):
        """
        Runs the chosen classifiers using TimeSeries Cross validation
        :return: 
        """
        X_train_resampled, y_train_resampled, X_test, y_test, X_train, y_train = self._get_data_for_classifier()

        kcv, time_cv = self._run_cross_validation()
        print "Cross Validation on Decision tree"
        clf = DecisionTreeClassifier(min_samples_split=50)
        cross_scores = cross_val_score(clf, X_train_resampled, y_train_resampled, cv = time_cv, scoring='recall')

        print cross_scores
        print "Mean Cross validation recall score = {}" .format(np.mean(cross_scores))

        print "Cross Validation on Majority Voting Ensemble"

        clf1 = DecisionTreeClassifier(min_samples_split=50)
        clf2 = RandomForestClassifier()

        clfV = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2)])

        cross_scores_ensemble = cross_val_score(clfV, X_train_resampled, y_train_resampled, cv=time_cv, scoring='recall')
        print cross_scores_ensemble
        print "Mean Cross validation recall score = {}" .format(np.mean(cross_scores_ensemble))

# if __name__ == '__main__':
#     c = Classifier()
#     c.run_neural_network()
