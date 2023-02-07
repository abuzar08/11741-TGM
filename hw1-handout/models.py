from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

import config
import retrieval
import utils
from utils import StatusCode


class Ranker(ABC):
    def __init__(self, numDocs, args, alpha = 0.8, r=None):
        '''
        DESCRIPTION
        ---
        Initializes the pageRanks algorithm object.
        ---
        INPUTS
        ---
            numDocs (int): number of documents in consideration
            args (argparse.parser): arguments
            alpha (float): dampening factor
            r (np.array, (numDocs,)): Initial Vector
        '''
        
        self.numDocs = numDocs
        self.alpha = 1-alpha if args.algo=="GPR" else alpha
        if args.load_saved_matrix:
            self.transitionMatrix = utils.loadTransitionMatrix(savePath=config.TRANSITION_PATH)
        else:
            self.transitionMatrix = utils.loadTransitionMatrix(fileName=args.transition, numDocs=self.numDocs)
        self.args = args
        self.p_0 = None
        self.r = r
            
        self.initializeAlgorithm(args)
    
    @abstractmethod
    def initializeAlgorithm(self, args):
        pass
    
    def getRawScores(self):
        return np.array(self.r)
    
    def step(self):
        '''
        Carries out one update step for the pageRanks algorithm.
        '''
        self.r = ((1-self.alpha)*self.transitionMatrix).T@self.r + self.alpha*self.p_0
    
    def run(self, eps=config.EPS):
        '''
        Runs the pagerank power iterations till convergence.
        ---
        INPUTS:
        eps(float): error tolerance for convergence.
        '''
        prev = None
        iters = 0
        
        while (prev is None or np.allclose(self.r, prev, rtol=eps) != True) and iters < config.MAX_ITERS:
            prev = self.r
            self.step()
            iters += 1
            print(f"Iteration: {iters}", end="\r")
        
        if iters == config.MAX_ITERS:
            print(f"Took more than {config.MAX_ITERS} steps")
            print("Bye bye!")
            return StatusCode.FAILURE
        
        else:
            print(f"Converged in {iters} iterations")
            return StatusCode.SUCCESS
    
    @abstractmethod
    def getRanks(self):
        pass

class pageRanks(Ranker):
    
    def initializeAlgorithm(self, args):
        self.p_0 = np.ones(self.numDocs) / self.numDocs
        if self.r is None:
            self.r = np.random.normal(0,1,size=self.numDocs)
    
    def getRanks(self, indriDocs, usr, qNo, scoringFunction = retrieval.base, queries=None):
        scores = self.getRawScores()
        docs   = indriDocs[usr][qNo]["docs"]
        
        rawScores = scores[docs]
        scores    = scoringFunction(rawScores, indriDocs[usr][qNo]["relevance"])
        indriDocs[usr][qNo]["scores"] = scores
        
        ranks = np.argsort(-scores)
        indriDocs[usr][qNo]["ranks"] = ranks
        
        return indriDocs
    
    def getAllRanks(self):
        return np.argsort(-self.getRawScores())


class pageRanksPersonalized(Ranker):
    
    def initializeAlgorithm(self, args):
        docTopics = utils.loadDocTopics(args.docTopics) # (numDocs,12)
        docTopics = docTopics.T # (12, numDocs)
        
        # normalized doctopics
        rowSums = np.array(docTopics.sum(axis=1))[:,0]
        rows,cols = docTopics.nonzero()
        docTopics.data = docTopics.data / rowSums[rows]
        
        self.p_0 = docTopics.T # (numDocs,12)
        if self.r is None:
            self.r   = np.random.normal(0,1,size=(self.numDocs, config.NUM_TOPICS))
        
    def getPersonalizedScores(self, topicDistribution):
        rawScores = self.getRawScores()
        personalizedScores = rawScores @ topicDistribution
        personalizedScores = personalizedScores.flatten()
        return personalizedScores
    
    def getRanks(self, indriDocs, queries, usr, qNo, scoringFunction = retrieval.base):
        topicDistribution = queries[usr][qNo]
        docs   = indriDocs[usr][qNo]["docs"]
        
        scores = self.getPersonalizedScores(topicDistribution)
        scores = scores[docs]
        scores    = scoringFunction(scores, indriDocs[usr][qNo]["relevance"])
        
        indriDocs[usr][qNo]["scores"] = scores
        
        ranks = np.argsort(-scores)
        indriDocs[usr][qNo]["ranks"] = ranks
        
        return indriDocs
        

def getRanker(args):
    if args.algo == "GPR":
        return pageRanks(config.NUM_DOCS, args=args)
    
    else:
        return pageRanksPersonalized(config.NUM_DOCS, args=args)
        