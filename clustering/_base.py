import logging

class ClusteringBase:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit_predict(self, data):
        raise NotImplementedError
