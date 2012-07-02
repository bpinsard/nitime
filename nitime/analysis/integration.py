import numpy as np

from nitime import descriptors as desc
from nitime import timeseries as ts
from nitime import algorithms as tsa

# To suppport older versions of numpy that don't have tril_indices:
from nitime.index_utils import tril_indices

from .base import BaseAnalyzer


class IntegrationAnalyzer(BaseAnalyzer):
    """Analyzer object for integration analysis."""

    def __init__(self, input=None, networks=None):
        """
        Parameters
        ----------
        
        input: Correlation Matrix object
           Containing the data to analyze.
           """
        self.networks = networks
        BaseAnalyzer.__init__(self, input)

#    def _entropy(m):
#        return 0.5*np.log(np.linalg.det(m))
    
    def _integration(self,m):
        return -0.5*np.log(np.linalg.det(m))

    @desc.setattr_on_read
    def integration(self):
        
        total = self._integration(self.input)
        intra = dict()
        if self.networks:
            for net,rois in self.networks.items():
                intra[net] = self._integration(self.input[rois,:][:,rois])
        inter = dict([(net,total-i) for net,i in intra.items()])

        return total, inter, intra
