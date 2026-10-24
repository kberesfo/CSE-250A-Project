import pandas as pd


class Discretizer:
    def __init__(self, n_bins: int): 
        self.n_bins = n_bins
        self.edges = {}

    def fit(self, train: pd.DataFrame, cols): 
        #TODO: ... # quantile edges from train only
        raise NotImplementedError
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        #TODO: ...   # same edges for val/test
        raise NotImplementedError
    