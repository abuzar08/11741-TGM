# FILE WITH DEFAULT CONFIGURATIONS

EPS  = 1e-8
SEED = 0

TRANSITION_PATH = "./data/sparseTransition.npz"
INDRI_PATH      = "./data/indri-lists"

NUM_TOPICS = 12
NUM_DOCS   = 81433
MAX_ITERS  = 1000

WS_WEIGHT = 0.5
SCORERS = ["NS", "WS", "CS"]
ALGOS = ["GPR", "QTSPR", "PTSPR"]

NUM_BINOMIAL_TRIALS = 100
