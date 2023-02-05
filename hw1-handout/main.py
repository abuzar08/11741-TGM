from utils import *

if __name__ == "__main__":
    
    args = getArgs()
    dbg = debugPrint(args.debug)
    transitionDict, transitionSparse, queryTopics, docTopics, userTopics = loadData(args, dbg)