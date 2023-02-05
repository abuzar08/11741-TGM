import numpy as np
import argparse
from collections import defaultdict
from utils import *

def loadData(args):
    '''
    Loads all data
    '''
    transitionDict, transitionSparse = loadTransitionMatrix(args.transition)
    dbg("Transition: ", len(transitionDict), transitionSparse.shape)
    
    queryTopics = loadQueryMatrix(args.queryTopics)
    dbg("Query: ", len(queryTopics))
    
    docTopics = loadQueryMatrix(args.docTopics)
    dbg("Doc Topics: ", len(docTopics))
    
    userTopics = loadUserMatrix(args.userTopics)
    dbg("User Topics: ", len(userTopics))

    dbg("Finished loading data")
    
    return transitionDict, transitionSparse, queryTopics, docTopics, userTopics

if __name__ == "__main__":
    
    args = getArgs()
    dbg = debugPrint(args.debug)
    transitionDict, transitionSparse, queryTopics, docTopics, userTopics = loadData(args)