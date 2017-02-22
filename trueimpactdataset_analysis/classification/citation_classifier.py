
import logging
import operator

import numpy
from sklearn.metrics import confusion_matrix

from trueimpactdataset_analysis.logutils import LogUtils

__author__ = 'robodasha'
__email__ = 'damirah@live.com'


class CitationClassifier(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def baseline(self, labels):
        """
        Get baseline accuracy -- classify everything as majority class
        :param labels: list of True/False values (True representing a seminal
                       paper and False a survey)
        :return: baseline accuracy
        """
        # this tells us how many true/false labels are there
        majority_class = sum(labels) * 2 > len(labels)
        correct_base = labels.count(majority_class)
        return correct_base / len(labels)

    def train(self, citations, labels, print_progress=True, optimal=False):
        """
        :param citations: list of number of citations for each paper
        :param labels: labels (True/False) showing which papers are seminal,
                       labels array has to have same length as citations array
        :param print_progress: whether method should log progress
        :param optimal:
        :return: tuple -- training accuracy, the threshold achieving
                 that accuracy and the confusion matrix for that threshold
        """
        if len(numpy.unique(labels)) <= 1 or len(citations) != len(labels):
            self._logger.warning(
                'Citations and labels arrays must be non-empty, have same '
                'length and contain at least one seminal and one survey paper')
            return None, None, None

        total = max(citations) + 1
        how_often = LogUtils.how_often(total)
        processed = 0

        self._logger.info('Will test {} thresholds'.format(total))

        accuracy = {}
        for threshold in range(max(citations) + 1):
            predicted = [
                True if cit >= threshold else False for cit in citations]
            correct = [
                predicted[i] == labels[i] for i in range(len(predicted))]
            accuracy[threshold] = sum(correct) / len(correct)
            processed += 1
            if print_progress and processed % how_often == 0:
                self._logger.debug(LogUtils.get_progress(processed, total))

        # sort by accuracy from highest to lowest
        sorted_accuracy = sorted(
            accuracy.items(), key=operator.itemgetter(1), reverse=True)
        accuracy_best = sorted_accuracy[0][1]
        # for the optimal model using all data we can pick any of the best
        # thresholds, as all will have the same performance
        if optimal:
            threshold_best = sorted_accuracy[0][0]
        # for the cross-validation setup we use the mean value
        # of all best thresholds
        else:
            threshold_best_all = [
                t for t, a in sorted_accuracy if a == accuracy_best]
            threshold_best = sum(threshold_best_all) / len(threshold_best_all)
        self._logger.info('Optimal threshold: {}, accuracy: {}'.format(
            threshold_best, accuracy_best))

        # get confusion matrix for the selected threshold
        pred = [True if cit >= threshold_best else False for cit in citations]
        cm = confusion_matrix(labels, pred)

        return accuracy_best, threshold_best, cm

    def test(self, citations, labels):
        """
        Leave-one-out cross-validation
        :param citations: list of number of citations for each paper
        :param labels: labels (True/False) showing which papers are seminal,
                       labels array has to have same length as citations array
        :return: tuple -- test accuracy, list of trained thresholds, confusion
                 matrix
        """
        if len(numpy.unique(labels)) <= 1 or len(citations) < 3 \
                or len(citations) != len(labels):
            self._logger.warning(
                'Citations and labels arrays must be non-empty, have same '
                'length and contain at least one seminal and one survey paper')
            return None, None, None

        total = len(citations)
        how_often = LogUtils.how_often(total)
        processed = 0

        self._logger.info('Will run {} iterations'.format(total))

        results = []
        true_labels = []
        thresholds = []
        for idx, cit in enumerate(citations):
            training_data = citations.copy()
            training_labels = labels.copy()
            del training_data[idx]
            del training_labels[idx]
            _, threshold, _ = self.train(
                training_data, training_labels, print_progress=False)
            if threshold is None:
                continue
            results.append(cit >= threshold)
            thresholds.append(threshold)
            true_labels.append(labels[idx])
            processed += 1
            if processed % how_often == 0:
                self._logger.debug(LogUtils.get_progress(processed, total))

        correct = [true_labels[i] == results[i] for i in range(len(results))]
        accuracy = sum(correct) / len(correct)
        cm = confusion_matrix(true_labels, results)
        self._logger.info('Test accuracy after {} iterations: {}'.format(
            len(citations), accuracy))

        return accuracy, thresholds, cm
