from collections import defaultdict
import argparse
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import os

def getArgs():
    '''
    Argparser. 
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition", type=str,default="./data/transition.txt",
                        help="path to transition matrix")
    parser.add_argument("--queryTopics", type=str,default="./data/query-topic-distro.txt",
                        help="path to query topic distribution matrix")
    parser.add_argument("--docTopics", type=str, default="./data/doc_topics.txt",
                        help="path to document topic matrix")
    parser.add_argument("--userTopics", type=str, default="./data/user-topic-distro.txt",
                        help="path to user-topic distribution matrix")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--algo", type=str, default="GPR", help="One of [GPR, QTSPR, PTSPR]")
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args


def loadTransitionMatrix(fileName: str, numDocs: int):
    '''
    Loads the sparse transition matrix.
    Converts it into a dictionary of int to list of int of transitions
    '''
    
    if os.path.exists("./data/sparseTransition.npz"):
        sparseMatrix = sp.load_npz("./data/sparseTransition.npz")
        return sparseMatrix
    
    data = np.loadtxt(fileName, dtype=np.int)
    matrix = defaultdict(lambda: [])
    for row in data:
        src, dst = row[0] - 1, row[1] - 1
        matrix[src].append(dst)
    
    row    = data[:, 0] - 1
    col    = data[:, 1] - 1
    values = data[:, 2]
    sparseMatrix = sp.csr_matrix((values, (row, col)), dtype=np.int8, shape=(numDocs, numDocs))
    
    rowSums = np.array(sparseMatrix.sum(axis=1))[:,0]
    rows,cols = sparseMatrix.nonzero()
    sparseMatrix.data = sparseMatrix.data / rowSums[rows]
    # for r,c in tqdm(zip(rows,cols), desc="Setting non Zeros"):
    #     sparseMatrix[r,c] = 1 / len(matrix[r])
    
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

def loadQueryMatrix(fileName: str):
    '''
    Loads the sparse query matrix.
    Converts it into a dictionary of int to dictionary of topic distribution vectors.
    '''
    data = np.loadtxt(fileName, dtype=object, delimiter=" ")
    matrix = defaultdict(lambda: {})
    for row in data:
        user, qNum = int(row[0])-1, int(row[1])-1
        vector = np.array([item.split(':')[1] for item in row[2:]], dtype = np.float)
        matrix[user][qNum] = vector
    return matrix

def loadDocTopics(fileName:str):
    '''
    Loads the sparse document-topic distribution.
    '''
    data = np.loadtxt(fileName, dtype=np.int, delimiter=" ")
    matrix = defaultdict(lambda: [])
    for row in data:
        doc, topic = row[0], row[1]
        matrix[doc].append(topic)
    numDocs = np.max(data[:,0])
    return matrix, numDocs

def loadUserMatrix(fileName: str): 
    '''
    Loads the sparse query matrix.
    Converts it into a dictionary of int to dictionary of topic distribution vectors.
    '''
    data = np.loadtxt(fileName, dtype=object, delimiter=" ")
    matrix = defaultdict(lambda: {})
    for row in data:
        user, qNum = int(row[0])-1, int(row[1])-1
        vector = np.array([item.split(':')[1] for item in row[2:]], dtype = np.float)
        matrix[user][qNum] = vector
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