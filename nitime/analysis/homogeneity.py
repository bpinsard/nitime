import numpy as np

from nitime import descriptors as desc
from nitime import timeseries as ts
from nitime import algorithms as tsa

from .base import BaseAnalyzer


class HomogeneityAnalyzer(BaseAnalyzer):
    """Analyzer object for homogeneity analysis."""

    def __init__(self, input=None):
        """
        Parameters
        ----------

        input: TimeSeries object
           Containing the data to analyze.
        """
        BaseAnalyzer.__init__(self, input)


    @desc.setattr_on_read
    def homogeneity(self):
        from bottleneck import rankdata
        ts=rankdata(self.input.data,axis=1)

        y = (ts - ts.mean(1))**2
        
        N,T = ts.shape
        K = (N**2*T*(T**2-1))/(12*(N-1))
        
        return 1-y.sum()/K
