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
    bonusProbs = (rawScores - np.min(rawScores)) / np.sum(rawScores - np.min(rawScores))
    bonusPoints = np.array([np.random.binomial(n=config.NUM_BINOMIAL_TRIALS,p=bonusProbs[i],size=1) for i in range(len(bonusProbs))]).flatten() / config.NUM_BINOMIAL_TRIALS
    scores = relevanceScores + bonusPoints
    return scores

def getScorer(args):
    if args.scorer == "NS":
        return base
    
    elif args.scorer == "WS":
        return weightedSum
    
    elif args.scorer == "CS":
        return customSum
    
    else:
        raise ValueError("--scorer should be one of [NS, WS, or CS]")