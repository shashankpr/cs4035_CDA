import numpy as np
import pandas as pd
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc


class ClassificationMetric(object):
    def __init__(self):
        pass

    def compute_precision_score(self, y_true, y_pred, average_type):
        p_score = precision_score(y_true, y_pred, average=average_type)

        logging.info("Precision Score: {}".format(p_score))

        return p_score

    def compute_recall_score(self, y_true, y_pred, average_type):
        r_score = recall_score(y_true, y_pred, average=average_type)

        logging.info("Recall Score: {}".format(r_score))
        return r_score

    def classification_report(self, y_true, y_pred):

        target_names = ['class 0', 'class 1']
        report = classification_report(y_true, y_pred, target_names)
        logging.info("Classification report:")
        logging.info(report)

        return report

    def get_confusion_matrix(self, y_true, y_pred):

        conf_matrix = confusion_matrix(y_true, y_pred)
        logging.info("Confusion Matrix: {}".format(conf_matrix))

        return conf_matrix

    def get_roc(self, y_true, y_pred_prob):

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        return roc_auc

    def get_full_report(self, y_true, y_pred, y_true_prob=None, y_pred_prob=None, roc_reqd=True):
        conf_matrix = self.get_confusion_matrix(y_true, y_pred)
        prec_score = self.compute_precision_score(y_true, y_pred, average_type='weighted')
        recal_score = self.compute_recall_score(y_true, y_pred, average_type='weighted')
        # c_report = self.c_metric.classification_report(y_test, predicted)

        print "Precision"
        print prec_score

        print "Recall"
        print recal_score

        conf_report = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
        print "Confusion matrix"
        print conf_report

        print "Classification Report"
        print classification_report(y_true, y_pred)

        if roc_reqd:
            print "ROC"
            roc = self.get_roc(y_true_prob, y_pred_prob)
            print roc