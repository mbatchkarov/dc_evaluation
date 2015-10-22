# coding=utf-8
from collections import defaultdict
import csv
import logging
import os

from numpy import count_nonzero
from sklearn.base import TransformerMixin


class FeatureVectorsCsvDumper(TransformerMixin):
    """
    Saves the vectorized input to file for inspection
    """

    def __init__(self, exp_name, cv_number=0, prefix='.'):
        self.exp_name = exp_name
        self.cv_number = cv_number
        self.prefix = prefix
        self._tranform_call_count = 0

    def _dump(self, X, y, file_name='dump.csv'):
        """
        The call order is
            1. training data
                1. fit
                2. transform
            2. test data
                1. transform

        We only want to dump after stages 1.2 and 2.1
        """
        matrix, vocabulary_ = X
        new_file = os.path.join(self.prefix, file_name)
        c = csv.writer(open(new_file, "w"))
        inverse_vocab = {index: word for (word, index) in vocabulary_.items()}
        v = [inverse_vocab[i] for i in range(len(inverse_vocab))]
        c.writerow(['id'] + ['target'] + ['total_feat_weight'] + ['nonzero_feats'] + v)
        for i in range(matrix.shape[0]):
            row = matrix.todense()[i, :].tolist()[0]
            vals = ['%1.2f' % x for x in row]
            c.writerow([i, y[i], sum(row), count_nonzero(row)] + vals)
        logging.info('Saved debug info to %s', new_file)

    def fit(self, X, y=None, **fit_params):
        self.y = y
        return self

    def transform(self, X):
        self._tranform_call_count += 1
        suffix = {1: 'tr', 2: 'ev'}
        if self._tranform_call_count == 2:
            self.y = defaultdict(str)

        if 1 <= self._tranform_call_count <= 2:
            self._dump(X, self.y,
                       file_name='PostVectDump_%s_%s-fold%r.csv' % (self.exp_name,
                                                                    suffix[self._tranform_call_count],
                                                                    self.cv_number))
        return X

    def get_params(self, deep=True):
        return {'cv_number': self.cv_number,
                'exp_name': self.exp_name}
