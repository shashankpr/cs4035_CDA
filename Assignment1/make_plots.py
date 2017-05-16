import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import itertools

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from classifiers import Classifier
from metrics import ClassificationMetric

c = Classifier()
m = ClassificationMetric()


class Plots(object):
    def __init__(self, X_train_resampled, y_train_resampled, X_test, y_test):
        self.X_train_resampled = X_train_resampled
        self.y_train_resampled = y_train_resampled
        self.X_test = X_test
        self.y_test = y_test

    def _plot_confusion_matrix(self, cm, classes,
                               normalize=False,
                               title='Confusion matrix',
                               cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            # print("Normalized confusion matrix")
        else:
            1  # print('Confusion matrix, without normalization')

        # print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plot_roc(self):
        """
        PLots ROC curves for given Dict of classifiers
        :return: 
        """
        tree = DecisionTreeClassifier(min_samples_split=50, random_state=0)

        rf = RandomForestClassifier(random_state=0)
        voting_clf = VotingClassifier(estimators=[('tree', tree), ('rf', rf)], voting='soft', weights=[3, 2])

        ada = AdaBoostClassifier(n_estimators=100, base_estimator=tree)

        colors = cycle(['cyan', 'indigo', 'seagreen'])
        lw = 2

        i = 0
        classifiers = {'DecisionTree': tree, 'Voting Ensemble': voting_clf, 'AdaBoost': ada}
        for name, clf in classifiers.items():
            probas_ = clf.fit(self.X_train_resampled, self.y_train_resampled).predict_proba(self.X_test)

            fpr, tpr, thresholds = roc_curve(self.y_test, probas_[:, 1])
            roc_auc = auc(fpr, tpr)

            plt.title('Receiver Operating Characteristic - %s' % name)
            plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([-0.1, 1.0])
            plt.ylim([-0.1, 1.01])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

    def plot_conf_matrix(self):
        """
        Plots confusion matrix for given dict of classifiers
        :return: 
        """
        tree = DecisionTreeClassifier(min_samples_split=50, random_state=0)

        rf = RandomForestClassifier(random_state=0)
        voting_clf = VotingClassifier(estimators=[('tree', tree), ('rf', rf)], voting='soft', weights=[3, 2])

        nn = MLPClassifier(solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,), random_state=0)
        X_pca_transf_train = c.pca_decompose(self.X_train_resampled)
        X_pca_transf_test = c.pca_decompose(self.X_test)

        ada = AdaBoostClassifier(n_estimators=100)

        classifiers = {'DecisionTree': tree, 'AdaBoost': ada, 'Voting Ensemble': voting_clf}
        # classifiers = {'Forest Ensemble': rf}
        for name, clf in classifiers.items():
            clf.fit(X_pca_transf_train, y_train_resampled)
            predict_ = clf.predict(X_pca_transf_test)

            cnf_matrix = m.get_confusion_matrix(y_test, predict_)
            np.set_printoptions(precision=2)

            class_names = [0, 1]
            plt.figure()
            self._plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix - %s' % name)
            plt.show()


if __name__ == '__main__':
    X_train_resampled, y_train_resampled, X_test, y_test = c.get_data_for_classifier()
    p = Plots(X_train_resampled, y_train_resampled, X_test, y_test)

    p.plot_roc()
    p.plot_conf_matrix()
