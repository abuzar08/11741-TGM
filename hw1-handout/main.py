import argparse
from datetime import datetime
import time

import numpy as np

import config
import models
import retrieval
import utils
from utils import StatusCode


def getArgs():
    '''
    Argparser. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", type=str,default="./data/transition.txt",
                        help="path to transition matrix")
    parser.add_argument("--load_saved_matrix", action="store_true", default=False)
    parser.add_argument("--queryTopics", type=str,default="./data/query-topic-distro.txt",
                        help="path to query topic distribution matrix")
    parser.add_argument("--docTopics", type=str, default="./data/doc_topics.txt",
                        help="path to document topic matrix")
    parser.add_argument("--userTopics", type=str, default="./data/user-topic-distro.txt",
                        help="path to user-topic distribution matrix")
    
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--algo", type=str, default="all", help="One of [GPR, QTSPR, PTSPR, all]")
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta",  type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--scorer", type=str, default="all", help="default All. To change, use NS or WS or CS")
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

def printTimeDiff(start, end, avgFactor = 1.0):
    delta = (end - start) / avgFactor
    deltaMinutes = delta//60
    deltaSeconds = delta%60
    print(f"Completed power iteration in {deltaMinutes}m and {deltaSeconds}s")

if __name__ == "__main__":
    
    args = getArgs()
    dbg = utils.debugPrint(args.debug)
    if args.algo not in ["GPR", "QTSPR", "PTSPR", "all"]:
        print("Invalid Algo specified. Should be one of GPR, QTSPR, PTSPR, or all")
        exit(1)
    
    if args.algo == "all":
        algorithms = config.ALGOS
    
    else:
        algorithms = [args.algo]
    
    for algo in algorithms:
        args.algo = algo
        argsScorer = args.scorer
        print("="*100)
        print(f"Algorithm: {args.algo}")

            
        np.random.seed(args.seed)
        
        
        
        ranker = models.getRanker(args)
        indriDocs = utils.loadIndri()
        
        rankerStartTime = time.time()
        status  = ranker.run(eps=config.EPS)
        rankerEndTime = time.time()
        
        print(f"Completed power iteration in {rankerEndTime - rankerStartTime:.3f} seconds.")
        print("="*100)
        
        # if status == StatusCode.FAILURE:
        #     print(f"Failed to converge in {config.MAX_ITERS} iterations")
        
        if args.algo == "GPR":
            queries = None
        
        elif args.algo == "QTSPR":
            queries = utils.loadQueries(args.queryTopics) # dictionary
            
        elif args.algo == "PTSPR":
            queries = utils.loadQueries(args.userTopics) # dictionary
        
        
        if args.scorer == "all":
            scorers = config.SCORERS
        else:
            scorers = [args.scorer]
        
        for scorer in scorers:
            args.scorer = scorer
            scoringFunction = retrieval.getScorer(args)
            print(f"> Retrieval function: {args.scorer}")
            
            retrievalStart = time.time()
            allData = []
            for usr in indriDocs:
                for qNo in indriDocs[usr]:
                    indriDocs = ranker.getRanks(indriDocs=indriDocs,
                                    usr= usr, qNo= qNo, 
                                    queries=queries,
                                    scoringFunction=scoringFunction)
                    data = utils.makeOuputFile(indriDocs, usr, qNo, args)
                    allData.extend(data)
            
            retrievalEnd = time.time()
            print(f"  Completed retrieval in {retrievalEnd - retrievalStart:.3f} seconds per query\n")
            
            if args.algo == "GPR":
                ranker.genFile()
            
            else:
                ranker.genFile(algo=args.algo, u=2, q=1, topicDistribution=queries[2][1])
            # print("-"*100)
        args.scorer = argsScorer
    # utils.writeOutput(allData, args)
            
        