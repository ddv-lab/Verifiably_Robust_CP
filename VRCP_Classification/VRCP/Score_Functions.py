import numpy as np
import bisect
from scipy.stats import rankdata, percentileofscore


# The HPS non-conformity score
def class_probability_score(probabilities, labels, u=None, all_combinations=False):

    # get number of points
    num_of_points = np.shape(probabilities)[0]

    # calculate scores of each point with all labels
    if all_combinations:
        scores = 1 - probabilities[:, labels]

    # calculate scores of each point with only one label
    else:
        scores = 1 - probabilities[np.arange(num_of_points), labels]

    # return scores
    return scores

class ranking_score(object):
    """
    Apply ranking transformation on base score.
    """
    def __init__(self, training_scores, base_score):
        self.ref_scores = sorted(training_scores.tolist())
        self.base_score = base_score
        self.score_func = lambda x: bisect.bisect(self.ref_scores, x) / len(self.ref_scores)
    def __call__(self, probabilities, labels, u, **kwargs):
        scores = self.base_score(probabilities, labels, u, **kwargs)
        quantile_score = np.frompyfunc(self.score_func, 1, 1)(scores)
        return quantile_score.astype(float)


class sigmoid_score(object):
    """
    Apply sigmoid transformation on base score.
    """
    def __init__(self, base_score, T=10, bias=0.5):
        self.base_score = base_score
        self.T = T
        self.bias = bias
    def __call__(self, probabilities, labels, u, **kwargs):
        scores = self.base_score(probabilities, labels, u, **kwargs)
        return 1 / (1 + np.exp(-self.T * (scores - self.bias)))