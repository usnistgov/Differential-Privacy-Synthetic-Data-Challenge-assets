"""
This file contains the implementation of the scoring metric so that you can use it locally
or reimplement it in other languages. See the problem description for an explanation of
the metric.
"""

import numpy as np
from scipy.spatial.distance import jensenshannon


class Deid2Metric:
    def __init__(self):
        self.threshold = 0.05
        self.misleading_presence_penalty = 0.2
        self.bias_penalty = 0.25
        self.allowable_raw_bias = 500

    @staticmethod
    def _normalize(arr):
        """ Turn raw counts into frequencies but handle all-zero arrays gracefully """
        if (arr > 0).any():
            return arr / arr.sum()
        return arr

    def _zero_below_threshold(self, arr):
        """ Take any entries that are below the threshold we care about and zero them out """
        return np.where(self._normalize(arr) >= self.threshold, arr, 0)

    def _penalty_components(self, actual, predicted):
        """ Score one row of counts for a particular (neighborhood, year, month) """
        if (actual == predicted).all():
            return 0.0, 0.0, 0.0

        # get the overall penalty for bias
        bias_mask = np.abs(actual.sum() - predicted.sum()) > self.allowable_raw_bias
        bias_penalty = self.bias_penalty if bias_mask.any() else 0

        # zero out entries below the threshold
        gt = self._zero_below_threshold(actual).ravel()
        dp = self._zero_below_threshold(predicted).ravel()

        # get the base Jensen Shannon distance; add a tiny bit of weight to each bin in order
        # to avoid all-zero arrays (and thus NaNs) without unduly influencing the distribution
        # induced by normalizing the arrays (dividing by sums); use base 2 to get a proper
        # spread of scores between [0, 1] instead of default base e which is more like [0, 0.83]
        #
        # ref: https://docs.scipy.org/doc/scipy-1.5.2/reference/generated/scipy.spatial.distance.jensenshannon.html
        jsd = jensenshannon(gt + 1e-9, dp + 1e-9, base=2)

        # get the overall penalty for the presence of misleading counts
        misleading_presence_mask = (gt == 0) & (dp > 0)
        misleading_presence_penalty = (
            misleading_presence_mask.sum() * self.misleading_presence_penalty
        )

        return jsd, misleading_presence_penalty, bias_penalty

    def _raw_row_scores(self, actual, predicted):
        n_rows, _n_incidents = actual.shape
        raw_penalties = np.zeros(n_rows, dtype=np.float)
        for i in range(n_rows):
            components_i = self._penalty_components(actual[i, :], predicted[i, :])
            raw_penalties[i] = np.sum(components_i)
        raw_scores = np.ones_like(raw_penalties) - raw_penalties
        return raw_scores

    def score(self, actual, predicted, return_individual_scores=False):
        # make sure the submitted values are proper
        assert np.isfinite(predicted).all()
        assert (predicted >= 0).all()

        # get all of the individual scores
        raw_scores = self._raw_row_scores(actual, predicted)

        # clip all the scores to [0, 1]
        scores = np.clip(raw_scores, a_min=0.0, a_max=1.0)

        # sum up the scores - a perfect score would be the length of the submission format
        overall_score = np.sum(scores)

        # in some uses (like visualization), it's useful to get the individual scores out too
        if return_individual_scores:
            return overall_score, scores

        return overall_score
