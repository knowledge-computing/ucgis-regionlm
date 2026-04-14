import logging

class DimensionReducerBase:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def fit_transform(self, data):
        raise NotImplementedError
