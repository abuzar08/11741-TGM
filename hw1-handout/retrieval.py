import numpy as np
import scipy.sparse as sp

def weightedSum(rankScores, relevanceScores, beta=0.5):
    score = beta*rankScores + (1-beta)*relevanceScores
    return score

def customSum(rankScores, relevanceScores)
    raise NotImplementedError