import time

import numpy as np

import config
import models
import retrieval
import utils
from utils import StatusCode

if __name__ == "__main__":
    
    args = utils.getArgs()
    dbg = utils.debugPrint(args.debug)
    if args.algo not in ["GPR", "QTSPR", "PTSPR", "all"]:
        print("Invalid Algo specified. Should be one of GPR, QTSPR, PTSPR, or all")
        exit(1)
    
    if args.algo == "all":
        '''
        If running all three algorithms
        '''
        algorithms = config.ALGOS
    
    else:
        algorithms = [args.algo]
    
    print("Normalizing Transition matrix...")
    transitionMatrix = utils.loadTransitionMatrix(args.transition, config.NUM_DOCS, config.TRANSITION_PATH)
    print()
    
    for algo in algorithms:
        args.algo = algo
        argsScorer = args.scorer
        print("="*100)
        print(f"Algorithm: {args.algo}")

        np.random.seed(args.seed)
        
        ranker = models.getRanker(args, transitionMatrix)
        indriDocs = utils.loadIndri()
        
        rankerStartTime = time.time()
        status  = ranker.run(eps=config.EPS)
        rankerEndTime = time.time()
        
        print(f"Completed power iteration in {rankerEndTime - rankerStartTime:.3f} seconds.")
        print("="*100)
        
        if status == StatusCode.FAILURE:
            # print(f"Failed to converge in {config.MAX_ITERS} iterations")
            pass
        
        if args.algo == "GPR":
            queries = None
        
        elif args.algo == "QTSPR":
            queries = utils.loadQueries(args.queryTopics) # dictionary
            
        elif args.algo == "PTSPR":
            queries = utils.loadQueries(args.userTopics) # dictionary
        
        
        if args.scorer == "all":
            '''
            If using all three scoring methods
            '''
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
        
            if not args.no_op:
                '''
                If required to create output file for trec_eval
                '''
                utils.writeOutput(allData, args)
        
        args.scorer = argsScorer
            
        