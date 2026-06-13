from pathlib import Path

# ── Project Paths ────────────────────────────────────────────
# Using pathlib.Path so this works on Windows, Mac, and Linux
# without worrying about forward/back slashes
ROOT_DIR   = Path(__file__).parent          # wherever config.py lives
DATA_DIR   = ROOT_DIR / "data"              # raw datasets go here
CKPT_DIR   = ROOT_DIR / "checkpoints"      # saved model weights
LOG_DIR    = ROOT_DIR / "logs"             # training logs / csvs
VOCAB_FILE = DATA_DIR / "vocab_map.json"   # our reduced vocabulary

# Create directories if they don't exist yet
# exist_ok=True means: don't crash if folder already exists
for _dir in [DATA_DIR, CKPT_DIR, LOG_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Tokenizer ────────────────────────────────────────────────
# We use BLOOM's tokenizer but only keep the 32,000 most
# frequent tokens from GSM8K. This keeps embedding table small.
BLOOM_MODEL_NAME = "bigscience/bloom-560m"
VOCAB_SIZE       = 32_000    # reduced from BLOOM's 250,880
MAX_SEQ_LEN      = 256       # max tokens per input sequence

# ── Model Architecture ───────────────────────────────────────
# These dimensions define the "thinking space" of the model.
#
# EXAMPLE: "If 3x + 9 = 21, find x"
#   After tokenization: ~12 tokens
#   After embedding: shape [12, 256]  ← each token is a 256-dim vector
#   After encoder:   shape [256]      ← whole sentence compressed to 1 vector
#
D_MODEL    = 256    # every vector in the system is 256-dimensional
N_HEADS    = 8      # attention splits 256 dims into 8 heads of 32 dims each
N_LAYERS   = 4      # 4 transformer layers in encoder
D_FFN      = 1024   # feed-forward hidden size (4× D_MODEL, standard ratio)
D_HEAD     = D_MODEL // N_HEADS   # = 32, dimension per attention head

# MLA (Multi-Head Latent Attention) compression dimensions
# Instead of projecting to D_MODEL directly, MLA compresses first
# KV compressed to 64 dims (saves 4× memory in KV cache)
# Q compressed to 96 dims (slightly larger, more expressiveness for queries)
D_LATENT_KV = 64
D_LATENT_Q  = 96

# ── JEPA Predictor ───────────────────────────────────────────
PRED_HIDDEN          = 1024   # MLP hidden layer width
N_LATENT_STEPS_TRAIN = 5      # teacher-forcing steps during training
N_LATENT_STEPS_INFER = 10     # free-running steps at inference

# VQ Codebook: 4096 discrete "reasoning concepts"
# Think of it as a dictionary of 4096 entries, each 256-dim
# After each predictor step, output snaps to nearest entry
VQ_CODEBOOK_SIZE  = 4096
VQ_COMMIT_WEIGHT  = 0.25   # how hard codebook chases encoder outputs
VQ_RESTART_THRESH = 10     # epochs before dead code gets reset

# ── Reconstruction Anchor ────────────────────────────────────
# λ_anchor increases over training: gently first, then fully
ANCHOR_WEIGHT_STAGE = {
    "early":  (1,  10, 0.1),   # (start_epoch, end_epoch, weight)
    "middle": (11, 30, 0.3),
    "late":   (31, 60, 0.5),
}

# ── VICReg Loss Weights ──────────────────────────────────────
# From the VICReg paper (Bardes et al., ICLR 2022)
# These are the recommended defaults. Don't change initially.
VICREG_INV   = 25.0   # invariance: prediction must match target
VICREG_VAR   = 25.0   # variance: each dim must have std >= GAMMA
VICREG_COV   = 1.0    # covariance: dims must be independent
VICREG_GAMMA = 1.0    # target standard deviation per dimension

# ── EMA (Exponential Moving Average) ─────────────────────────
# Target encoder lags behind context encoder.
# τ close to 1.0 = slow updates = large gap = strong learning signal
EMA_TAU_EARLY = 0.990   # epochs 1-20:  large gap, fast learning
EMA_TAU_MID   = 0.995   # epochs 21-50: medium gap, refinement
EMA_TAU_LATE  = 0.999   # epochs 51+:   small gap, fine polish

# ── Uncertainty Estimator ────────────────────────────────────
UNC_VAR_THRESHOLD   = 0.10   # min variance of final latent state
UNC_DRIFT_THRESHOLD = 1.0    # max avg step-to-step drift
UNC_W1 = 0.6   # weight for variance check (more important)
UNC_W2 = 0.4   # weight for stability check

# ── Training: Stage 1 (Encoder) ──────────────────────────────
S1_LR           = 1e-4
S1_LR_MIN       = 1e-6
S1_WEIGHT_DECAY = 0.01
S1_GRAD_CLIP    = 1.0
S1_BATCH_SIZE   = 16     # safe for laptop CPU and Kaggle GPU
S1_EPOCHS       = 100

# ── Training: Stage 2 (Predictor) ────────────────────────────
S2_LR           = 5e-5   # lower than Stage 1: encoder already stable
S2_LR_MIN       = 5e-7
S2_WEIGHT_DECAY = 0.01
S2_GRAD_CLIP    = 1.0
S2_BATCH_SIZE   = 16
S2_EPOCHS       = 60

# ── Training: Stage 3 (Decoder) ──────────────────────────────
S3_LR           = 1e-4
S3_LR_MIN       = 1e-6
S3_WARMUP_EPOCHS = 5     # ramp up LR to prevent early instability
S3_WEIGHT_DECAY = 0.01
S3_GRAD_CLIP    = 1.0
S3_BATCH_SIZE   = 8      # decoder is larger, needs more memory
S3_EPOCHS       = 50

# ── Inference ────────────────────────────────────────────────
TEMPERATURE     = 0.7    # sampling temperature: 0=greedy, 1=random
MAX_GEN_TOKENS  = 128    # max output tokens before forced stop

# ── Alarm Thresholds ─────────────────────────────────────────
# These trigger warnings during training
ALARM_VAR_HIGH    = 0.5    # L_var > this → collapse starting
ALARM_STD_LOW     = 0.05   # mean embedding std < this → STOP
ALARM_LOSS_RISE   = 5      # consecutive loss increases → reduce LR
ALARM_ANCHOR_HIGH = 1.5    # L_anchor > this after ep30 → raise λ
ALARM_CODEBOOK    = 200    # codes used < this → random restart

# ── Reproducibility ──────────────────────────────────────────
RANDOM_SEED = 42   # used everywhere: data splits, torch, numpy