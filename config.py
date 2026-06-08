# config.py - ALL hyperparameters in one place

# What is a hyperparameter?
# It's a setting YOU choose before training starts.
# The model does NOT learn these. YOU decide them.
# Think of it like oven temperature in baking.
# The cake doesn't choose the temperature - you do.

# ─── Model Architecture ───
VOCAB_SIZE = 32000          # How many unique "words" the model knows
D_MODEL = 256               # Size of each meaning vector (the "brain width")
N_HEADS = 8                 # How many "eyes" look at the input simultaneously
N_KV_HEADS = 2              # How many key/value groups (for GQA efficiency)
N_LAYERS = 4                # How many transformer layers (depth of thinking)
D_FFN = 1024                # Width of the feed-forward network (4× d_model)
MAX_SEQ_LEN = 256           # Maximum input length in tokens
DROPOUT = 0.1               # Randomly ignore 10% of signals (prevents memorization)

# ─── Memory System ───
MEMORY_DIM = 256            # Size of stored memory vectors
MAX_MEMORIES = 10000        # Maximum entries in ChromaDB
DECAY_RATE = 0.05           # How fast memories fade (5% per epoch)
DECAY_THRESHOLD = 0.05      # Below this score → delete the memory
RETRIEVAL_BOOST = 0.30      # Score boost when a memory is retrieved
TOP_K_MEMORIES = 5          # How many memories to retrieve per query

# ─── Training ───
BATCH_SIZE = 16             # How many examples to process at once
LEARNING_RATE = 3e-4        # How big each learning step is (0.0003)
EMA_TAU = 0.99              # How slowly the target encoder follows (99% old + 1% new)
VICREG_INVARIANCE = 25.0    # Weight for "predictions should match targets"
VICREG_VARIANCE = 25.0      # Weight for "vectors should be spread out"
VICREG_COVARIANCE = 1.0     # Weight for "dimensions should be independent"

# ─── Training Stages ───
STAGE1_EPOCHS = 100         # Encoder training epochs
STAGE2_EPOCHS = 60          # Predictor training epochs
STAGE3_EPOCHS = 50          # Decoder training epochs

# ─── Predictor ───
PREDICTOR_STEPS = 10        # How many reasoning steps in latent space

# ─── File Paths ───
CHECKPOINT_DIR = "checkpoints"
MEMORY_DB_DIR = "memory_db"
RESULTS_DIR = "results"
DATA_DIR = "data"

