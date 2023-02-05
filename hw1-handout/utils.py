from collections import defaultdict
import argparse
import numpy as np
import scipy.sparse as sp

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
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args


def loadTransitionMatrix(fileName: str):
    '''
    Loads the sparse transition matrix.
    Converts it into a dictionary of int to list of int of transitions
    '''
    data = np.loadtxt(fileName, dtype=np.int)
    
    row    = data[:, 0]
    col    = data[:, 1]
    values = data[:, 2]
    sparseMatrix = sp.csr_matrix((values, (row, col)), dtype=np.int8)
    
    matrix = defaultdict(lambda: [])
    for row in data:
        src, dst = row[0], row[1]
        matrix[src].append(dst)
    return matrix, sparseMatrix

def loadQueryMatrix(fileName: str):
    '''
    Loads the sparse query matrix.
    Converts it into a dictionary of int to dictionary of topic distribution vectors.
    '''
    data = np.loadtxt(fileName, dtype=object, delimiter=" ")
    matrix = defaultdict(lambda: {})
    for row in data:
        user, qNum = row[0], row[1]
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
    
    return matrix

def loadUserMatrix(fileName: str): 
    '''
    Loads the sparse query matrix.
    Converts it into a dictionary of int to dictionary of topic distribution vectors.
    '''
    data = np.loadtxt(fileName, dtype=object, delimiter=" ")
    matrix = defaultdict(lambda: {})
    for row in data:
        user, qNum = row[0], row[1]
        vector = np.array([item.split(':')[1] for item in row[2:]], dtype = np.float)
        matrix[user][qNum] = vector
    return matrix
        
def loadData(args, dbg):
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

class debugPrint:
    def __init__(self, flag):
        self.flag = flag
    
    def __call__(self, *args):
        if self.flag:
            print(*args)