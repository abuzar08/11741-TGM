import numpy as np
import scipy.sparse as sp

import config


def base(rawScores, relevanceScores=None):
    return rawScores

def weightedSum(rawScores, relevanceScores, weight = config.WS_WEIGHT):
    score = weight*rawScores + (1-weight)*relevanceScores
    return score

def customSum(rawScores, relevanceScores):
    score = 0.9*rawScores + 0.1*relevanceScores
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