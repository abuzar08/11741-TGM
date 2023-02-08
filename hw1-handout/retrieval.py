import numpy as np
import scipy.sparse as sp

import config


def base(rawScores, relevanceScores=None):
    return rawScores

def weightedSum(rawScores, relevanceScores, weight = config.WS_WEIGHT):
    score = weight*rawScores + (1-weight)*relevanceScores
    return score

def customSum(rawScores, relevanceScores):
    ranksRaw = np.argsort(np.argsort(rawScores))
    ranksRel = np.argsort(np.argsort(relevanceScores))
    averageRank = 0.05*ranksRaw + 0.95*ranksRel
    score = averageRank
    return score

def getScorer(args):
    if args.scorer == "NS":
        return base
    
    elif args.scorer == "WS":
        return weightedSum
    
    elif args.scorer == "CS":
        return customSum
    
    else:
        raise ValueError("--scorer should be one of [NS, WS, or CS]")