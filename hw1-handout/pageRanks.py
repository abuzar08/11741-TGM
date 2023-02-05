import numpy as np
import scipy.sparse as sp


def getTeleportationMatrix(numDocs):
    '''
    Gets teleportation matrix: p_zero @ 1.T
    '''
    p_zero =  np.ones(numDocs) / numDocs
    ones   =  np.ones(numDocs)
    return p_zero, ones