import argparse
import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import config


class StatusCode(enumerate):
    SUCCESS = 0
    FAILURE = 1


def loadTransitionMatrix(fileName: str=None, numDocs: int = 0, savePath: str=None):
    '''
    Loads the sparse transition matrix.
    Converts it into a dictionary of int to list of int of transitions
    '''
    
    if savePath is not None:
        if os.path.exists(savePath):
            sparseMatrix = sp.load_npz("./data/sparseTransition.npz")
            return sparseMatrix
        else:
            print(f"No file exists at {savePath}")
            raise FileNotFoundError
    
    if numDocs == 0:
        print("ERROR! Give numDocs please!")
        exit(1)
    
    data = np.loadtxt(fileName, dtype=np.int)
    row    = data[:, 0] - 1
    col    = data[:, 1] - 1
    values = data[:, 2]
    sparseMatrix = sp.csr_matrix((values, (row, col)), dtype=np.int8, shape=(numDocs, numDocs))
    
    # Normalizing non-zero values
    rowSums = np.array(sparseMatrix.sum(axis=1))[:,0]
    rows,cols = sparseMatrix.nonzero()
    sparseMatrix.data = sparseMatrix.data / rowSums[rows]
    
    # Normalizing zero values
    # lil matrix is faster when changing sparsity of csr
    sparseMatrix = sparseMatrix.tolil()
    for r in tqdm(range(numDocs), desc="Setting zeros"):
        if rowSums[r] == 0:
            rowVector = np.ones(numDocs, dtype=np.float64)
            rowVector[r] = 0
            rowVector = rowVector / (numDocs-1)
            sparseMatrix[r] = rowVector    
            
    # # convert back to csr matrix
    sparseMatrix = sparseMatrix.tocsr()
    # assert sparseMatrix.sum(axis=1).sum() == numDocs
    sp.save_npz("./data/sparseTransition.npz", sparseMatrix)
    return sparseMatrix

def loadQueries(fileName):
    '''
    Loads the Queries.
    Converts it into a dictionary of int to dictionary of topic distribution vectors.
    '''
    data = np.loadtxt(fileName, dtype=object, delimiter=" ")
    matrix = defaultdict(lambda: {})
    for row in data:
        user, qNum = int(row[0]), int(row[1])
        vector = np.array([item.split(':')[1] for item in row[2:]], dtype = np.float)
        matrix[user][qNum] = vector
    return matrix

def loadDocTopics(fileName:str):
    '''
    Loads the sparse document-topic distribution.
    '''
    data = np.loadtxt(fileName, dtype=np.int, delimiter=" ")
    numDocs = np.max(data[:,0])
    assert numDocs == config.NUM_DOCS
    row = data[:,0] - 1
    col = data[:,1] - 1
    values = np.ones(data.shape[0], dtype=np.int8)
    matrix = sp.csr_matrix((values, (row, col)), dtype=np.int8, shape=(numDocs, config.NUM_TOPICS))
    return matrix
        
def loadData(args, dbg):
    '''
    Loads all data
    '''
    # 1: Doc topics
    docTopics, numDocs = loadDocTopics(args.docTopics)
    dbg("Doc Topics: ", len(docTopics))
    dbg("Num Docs: ", numDocs)
    
    transitionSparse = loadTransitionMatrix(args.transition, numDocs)
    dbg("Transition: ", transitionSparse.shape)
    
    queryTopics = loadQueryMatrix(args.queryTopics)
    dbg("Query: ", len(queryTopics))
    
    userTopics = loadUserMatrix(args.userTopics)
    dbg("User Topics: ", len(userTopics))

    dbg("Finished loading data")
    
    return transitionSparse, queryTopics, docTopics, numDocs, userTopics

class debugPrint:
    def __init__(self, flag):
        self.flag = flag
    
    def __call__(self, *args):
        if self.flag:
            print(*args)

def loadIndri():
    '''
    DESCRIPTION
        Loads documents, relevance Scores, and ranks from indri-lists as a dictionary
    ---
    INPUTS
    ---
        queries (dict{ user(int): dict{query(int): np.array()}}): Queries loaded from loadQueries()
    '''
    indriDocs = defaultdict(lambda: defaultdict(lambda: {}))
    filesNames     = os.listdir(config.INDRI_PATH)
    
    for filename in filesNames:
        q_id = filename.split('.')[0]
        usr, qNo = q_id.split('-')
        usr, qNo = int(usr), int(qNo)
        
        path = os.path.join(config.INDRI_PATH, filename)
        data = np.loadtxt(path, dtype=object, delimiter=" ")
        
        docs      = data[:, 2].astype(np.int)
        relevance = data[:, 4].astype(np.float)
        
        indriDocs[usr][qNo]["docs"]      = docs
        indriDocs[usr][qNo]["relevance"] = relevance
        
    return indriDocs