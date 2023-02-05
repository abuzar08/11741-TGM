from utils import *

if __name__ == "__main__":
    
    args = getArgs()
    dbg = debugPrint(args.debug)
    transitionSparse, queryTopics, docTopics, numDocs, userTopics = loadData(args, dbg)
    algo = args.algo
    
    if algo not in ["GPR", "QTSPR", "PTSPR"]:
        print("Invalid Algo specified. Should be one of GPR, QTSPR, PTSPR.")
        exit(1)
    
    dbg("Algorithm: ",algo)