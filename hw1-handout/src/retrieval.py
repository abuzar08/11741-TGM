import numpy as np

import config


def base(rawScores, relevanceScores=None):
    '''
    Do nothing.
    '''
    return rawScores

def weightedSum(rawScores, relevanceScores, weight = config.WS_WEIGHT):
    '''
    Soft OR of pagerank scores and relevance scores.
    '''
    score = weight*rawScores + (1-weight)*relevanceScores
    return score

def customSum(rawScores, relevanceScores):
    '''
    
    '''
    normalizedRaw = (rawScores - np.min(rawScores)) / np.sum(rawScores - np.min(rawScores))
    argsorted = np.argsort(rawScores)
    numScores = len(argsorted)
    normalizedRaw[argsorted[:(98*numScores//100)]] = 0
    score = relevanceScores + normalizedRaw
    # score = 0.9*rawScores + 0.1*relevanceScores
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