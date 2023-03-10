from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp

import config
import retrieval
import utils
from utils import StatusCode


class Ranker(ABC):
    def __init__(self, numDocs, args, transitionMatrix, r=None):
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
        self.alpha = 1-args.alpha if args.algo=="GPR" else args.alpha
        self.transitionMatrix = transitionMatrix
        
        if args.algo != "GPR":
            self.beta  = args.beta
            self.gamma = args.gamma
            assert self.alpha + self.beta + self.gamma == 1.0, f"Non convex weights"
            
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
            # print(f"\tTook more than {config.MAX_ITERS} steps")
            # print("Bye bye!")
            return StatusCode.FAILURE
        
        else:
            print(f"Converged in {iters} iterations")
            return StatusCode.SUCCESS
    
    @abstractmethod
    def getRanks(self):
        pass
    
    @abstractmethod
    def genFile(self):
        pass

class pageRanks(Ranker):
    '''
    GPR
    '''
    def initializeAlgorithm(self, args):
        '''
        Initializes GPR
        '''
        self.p_0 = np.ones(self.numDocs) / self.numDocs
        self.alpha = 1-args.alpha
        if self.r is None:
            # self.r = np.random.normal(0,1,size=self.numDocs)
            self.r = np.ones(self.numDocs) / self.numDocs
    
    def getRanks(self, indriDocs, usr, qNo, scoringFunction = retrieval.base, queries=None):
        '''
        Takes indriDocs, user, query-number, and scoring function to 
        calculate final scores and ranks of each document in indrilists.
        '''
        scores = self.getRawScores()
        assert np.allclose(scores.sum(), 1.0, rtol=1e-4)
        docs   = indriDocs[usr][qNo]["docs"]
        
        rawScores = scores[docs-1]
        scores    = scoringFunction(rawScores, indriDocs[usr][qNo]["relevance"])
        indriDocs[usr][qNo]["scores"] = scores
        
        positions = np.argsort(np.argsort(-scores)) # get positions of document in a sorted array
        indriDocs[usr][qNo]["positions"] = positions
        
        return indriDocs
    
    def getAllRanks(self):
        return np.argsort(-self.getRawScores())
    
    def genFile(self, *args, **kwargs):
        '''
        Generates sample file for submission.
        '''
        lines = []
        for i,r in enumerate(self.r):
            lines.append(f"{i+1} {r}\n")
        
        with open("../GPR.txt", "w") as f:
            f.writelines(lines)


class pageRanksPersonalized(Ranker):
    '''
    TSPR
    '''
    def initializeAlgorithm(self, args):
        '''
        Initializes TSPR algorithm.
        '''
        docTopics = utils.loadDocTopics(args.docTopics) # (numDocs,12)
        docTopics = docTopics.T # (12, numDocs)
        
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        
        assert self.alpha + self.beta + self.gamma == 1.0, f"Non-convex weights"
        
        # normalized doctopics
        rowSums = np.sum(docTopics, axis=1, keepdims=True)
        docTopics = docTopics / rowSums
        
        self.p_t = docTopics.T # (numDocs,12)
        self.p_0 = np.ones((self.numDocs, config.NUM_TOPICS)) / self.numDocs
        if self.r is None:
            # self.r   = np.random.normal(0,1,size=(self.numDocs, config.NUM_TOPICS))
            self.r   = np.ones((self.numDocs, config.NUM_TOPICS)) / self.numDocs
    
    def step(self):
        '''
        Carries out one update step for the pageRanks algorithm.
        '''
        self.r = ((self.alpha)*self.transitionMatrix).T@self.r + self.beta*self.p_t + self.gamma*self.p_0
        
    def getPersonalizedScores(self, topicDistribution):
        rawScores = self.getRawScores()
        personalizedScores = rawScores @ topicDistribution
        personalizedScores = personalizedScores.flatten()
        assert np.allclose(personalizedScores.sum(), 1.0, rtol=1e-4)
        return personalizedScores
    
    def getRanks(self, indriDocs, queries, usr, qNo, scoringFunction = retrieval.base):
        '''
        Takes indriDocs, queries, user, query-number, and scoring function to 
        calculate final scores and ranks of each document in indrilists.
        '''
        topicDistribution = queries[usr][qNo]
        docs   = indriDocs[usr][qNo]["docs"]
        
        scores = self.getPersonalizedScores(topicDistribution)
        scores = scores[docs-1]
        scores    = scoringFunction(scores, indriDocs[usr][qNo]["relevance"])
        
        indriDocs[usr][qNo]["scores"] = scores
        
        positions = np.argsort(np.argsort(-scores)) # get positions of document in a sorted array
        indriDocs[usr][qNo]["positions"] = positions
        
        return indriDocs
    
    def genFile(self, algo, u,q, topicDistribution):
        '''
        Generate sample file for submission
        '''
        filename = f"../{algo}-U{u}Q{q}.txt"
        convergedValues = np.array(self.r @ topicDistribution)
        convergedValues = convergedValues.flatten()
        lines = []
        for i,r in enumerate(convergedValues):
            lines.append(f"{i+1} {r}\n")
        
        with open(filename, "w") as f:
            f.writelines(lines)
            
        

def getRanker(args, transitionMatrix):
    if args.algo == "GPR":
        return pageRanks(config.NUM_DOCS, args=args, transitionMatrix=transitionMatrix)
    
    else:
        return pageRanksPersonalized(config.NUM_DOCS, args=args, transitionMatrix=transitionMatrix)
        