import argparse

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
    parser.add_argument("--algo", type=str, default="GPR", help="One of [GPR, QTSPR, PTSPR]")
    parser.add_argument("--seed", type=int, default=config.SEED)
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args

if __name__ == "__main__":
    
    args = getArgs()
    dbg = utils.debugPrint(args.debug)
        
    np.random.seed(args.seed)
    
    if args.algo not in ["GPR", "QTSPR", "PTSPR"]:
        print("Invalid Algo specified. Should be one of GPR, QTSPR, PTSPR.")
        exit(1)
    
    dbg("Algorithm: ",args.algo)
    ranker = models.getRanker(args)
    indriDocs = utils.loadIndri()
    
    status  = ranker.run(eps=config.EPS)
    if status == StatusCode.FAILURE:
        print(f"Failed to converge in {config.MAX_ITERS} iterations")
    
    if args.algo == "GPR":
        queries = None
        ranks = ranker.getRanks()
        order = np.argsort(-ranks)
        print(order[:20])
    
    else:
        if args.algo == "QTSPR":
            queries = utils.loadQueries(args.queryTopics) # dictionary
        
        elif args.algo == "PTSPR":
            queries = utils.loadQueries(args.userTopics) # dictionary
        
        for usr in indriDocs:
            for qNo in indriDocs[usr]:
                indriDocs = ranker.getRanks(indriDocs=indriDocs,
                                usr= usr, qNo= qNo, 
                                queries=queries,
                                scoringFunction=retrieval.base)
        