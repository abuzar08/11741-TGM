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
    
    # sp.save_npz("./data/sparseTransition.npz", sparseMatrix)
    return sparseMatrix

def loadQueries(fileName):
    '''
    Loads the Queries.
    Converts it into a dictionary of int to dictionary of topic distribution vectors.
    '''
    data = np.loadtxt(fileName, dtype=object, delimiter=" ")
    queries = defaultdict(lambda: {})
    for row in data:
        user, qNum = int(row[0]), int(row[1])
        vector = np.array([item.split(':')[1] for item in row[2:]], dtype = np.float)
        queries[user][qNum] = vector
    return queries

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
    return matrix.toarray()

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
    indriDocs  = defaultdict(lambda: defaultdict(lambda: {}))
    filesNames = os.listdir(config.INDRI_PATH)
    
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

def makeOuputFile(indriDocs, usr, qNo, args):
    '''
    Generates the output data for trec_evaluation.
    '''
    q_id = f"{usr}-{qNo}"
    info = indriDocs[usr][qNo]
    data = []
    run_id = f"{args.algo}-{args.scorer}"
    for i in range(len(indriDocs[usr][qNo]["docs"])):
        document = info["docs"][i]
        # position     = info["positions"][i]+1
        score    = info["scores"][i]
        row = (q_id, "Q0", document, score, run_id)
        data.append(row)
    
    data.sort(key=lambda x: -x[3])
    return data

def printOutputData(data, n = 10):
    '''
    Prints the first n trec_eval output data rows for sanity checks.
    '''
    for i in range(n):
        print(data[i])

def writeOutput(data, args):
    '''
    Creates trec_eval submission files from data created by makeOutputFile()
    '''
    run_id = data[0][-1]
    filename = f"{run_id}.txt"
    writeLines = []
    rank = 0
    last_id = None
    for i,row in enumerate(data):
        if last_id!=row[0]:
            last_id = row[0]
            rank = 1
        line = f"{row[0]} {row[1]} {row[2]} {rank} {row[3]} {row[4]}\n"
        writeLines.append(line)
        rank += 1
    
    with open(filename, 'w') as f:
        f.writelines(writeLines)
        

def getArgs():
    '''
    Argparser. 
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--transition", type=str,default="./data/transition.txt",
                        help="path to transition matrix")

    parser.add_argument("--load_saved_matrix", action="store_true", default=False,
                        help="If set, load the saved normalized transition matrix")

    parser.add_argument("--queryTopics", type=str,default="./data/query-topic-distro.txt",
                        help="path to query topic distribution matrix")

    parser.add_argument("--docTopics", type=str, default="./data/doc_topics.txt",
                        help="path to document topic matrix")

    parser.add_argument("--userTopics", type=str, default="./data/user-topic-distro.txt",
                        help="path to user-topic distribution matrix")
    
    parser.add_argument("--debug", action='store_true', default=False, 
                        help="Set debug mode")

    parser.add_argument("--algo", type=str, default="all", help="One of [GPR, QTSPR, PTSPR, all]")

    parser.add_argument("--seed", type=int, default=config.SEED, 
                        help="Set random seed")

    parser.add_argument("--alpha", type=float, default=0.8, 
                        help="Alpha parameter (dampening factor for transition Matrix")

    parser.add_argument("--beta",  type=float, default=0.13,
                        help="Beta parameter (dampening factor for topic-based probability vector, p_t)")

    parser.add_argument("--gamma", type=float, default=0.07,
                        help="Gamma parameter (dampening factor for initial p_0 vector")
    
    parser.add_argument("--scorer", type=str, default="all", 
                        help="default All. To change, use NS or WS or CS")
    
    parser.add_argument("--no_op", action='store_true', default=False,
                        help="Set to BLOCK creation of output files for trec_eval.")
    
    
    args = parser.parse_args()
    print(f"args: {vars(args)}")
    return args
    
        