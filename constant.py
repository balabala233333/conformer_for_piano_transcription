import dataclasses
import os

import torch

WORKSPACE = r"./workspace"
DATASET_NAME = r"maestro-v3.0.0"

PEDAL_EXTEND = True

# =====================this is the config of transcription=========================
CONV_SUBSAMPLE = False
BLANK_ID = 0
INPUTS_LENGTH = 512
MUSICFORMER_TARGETS_LENGTH=1024
TARGETS_LENGTH = 2048
MAX_NUM_CACHED_FRAMES = 2000
NUM_ENCODER_LAYERS = 16
NUM_ATTENTION_HEADS = 4
ENCODER_INPUT_DIM = 512
DECODER_INPUT_DIM = 512
NUM_DECODER_LAYERS = 16
NUM_DECODER_ATTENTION_HEADS = 4
BATCH_SIZE = 15
BEGIN_NOTE = 22
CLASSES_NUM = 88
CHECKPOINT_PATH = os.path.join(WORKSPACE, "conformer")
if not os.path.exists(CHECKPOINT_PATH):
    os.mkdir(CHECKPOINT_PATH)
# 181
TEST_NOTE_CHECKPOINT_NAME = r"conformer/conformer_las_399"
#BEST 232
#BEST 241
#BEST 156
#BEST 347
#BEST 216
#BEST 343
TEST_NOTE_CHECKPOINT_PATH = os.path.join(WORKSPACE, TEST_NOTE_CHECKPOINT_NAME)

TEST_PEDAL_CHECKPOINT_NAME = r"conformer/conformer_pedal_10"
TEST_PEDAL_CHECKPOINT_PATH = os.path.join(WORKSPACE, TEST_PEDAL_CHECKPOINT_NAME)


TEST_NOTE_PEDAL_CHECKPOINT_NAME = r""
TEST_NOTE_PEDAL_CHECKPOINT_PATH = os.path.join(WORKSPACE,TEST_NOTE_PEDAL_CHECKPOINT_NAME)

DEVICE = torch.device("cpu")

LSTM_LAYERS = 1
@dataclasses.dataclass
class TokenConfig:
    inputs_length: int = INPUTS_LENGTH
    targets_length: int = TARGETS_LENGTH
    max_num_cached_frames: int = MAX_NUM_CACHED_FRAMES
    num_encoder_layers: int = NUM_ENCODER_LAYERS
    num_encoder_attention_heads: int = NUM_ATTENTION_HEADS
    encoder_input_dim: int = ENCODER_INPUT_DIM
    decoder_input_dim: int = DECODER_INPUT_DIM
    num_decoder_layers: int = NUM_DECODER_LAYERS
    num_decoder_attention_heads: int = NUM_DECODER_ATTENTION_HEADS
    checkpoint_path: str = CHECKPOINT_PATH
    device: torch.device = DEVICE
    begin_note = BEGIN_NOTE
    classes_num = CLASSES_NUM


# ====================the config of dataset path=======================
BASE_MAESTROV3_PATH = os.path.join(WORKSPACE, "maestro-v3.0.0")
BASE_MAESTROV2_PATH = os.path.join(WORKSPACE, "maestro-v2.0.0")
BASE_MAESTROV1_PATH = os.path.join(WORKSPACE, "maestro-v1.0.0")
CACHE_NAME = "cache"
PEDAL_CACHE_NAME = "cache_pedal"
PEDAL_NOTE_CACHE_NAME = "cache_pedal_note"
BASE_CHCHE_PATH = os.path.join(WORKSPACE, CACHE_NAME)
BASE_PEDAL_CACHE_PATH = os.path.join(WORKSPACE, PEDAL_CACHE_NAME)
BASE_PEDAL_NOTE_CACHE_PATH = os.path.join(WORKSPACE, PEDAL_NOTE_CACHE_NAME)

BASE_MAESTROV3_PEDAL_NOTE_CACHE_PATH = os.path.join(BASE_PEDAL_NOTE_CACHE_PATH, "maestro-v3.0.0")
BASE_MAESTROV2_PEDAL_NOTE_CACHE_PATH = os.path.join(BASE_PEDAL_NOTE_CACHE_PATH, "maestro-v2.0.0")
BASE_MAESTROV1_PEDAL_NOTE_CACHE_PATH = os.path.join(BASE_PEDAL_NOTE_CACHE_PATH, "maestro-v1.0.0")

BASE_MAESTROV3_PEDAL_CACHE_PATH = os.path.join(BASE_PEDAL_CACHE_PATH, "maestro-v3.0.0")
BASE_MAESTROV2_PEDAL_CACHE_PATH = os.path.join(BASE_PEDAL_CACHE_PATH, "maestro-v2.0.0")
BASE_MAESTROV1_PEDAL_CACHE_PATH = os.path.join(BASE_PEDAL_CACHE_PATH, "maestro-v1.0.0")

BASE_MAESTROV3_CACHE_PATH = os.path.join(BASE_CHCHE_PATH, "maestro-v3.0.0")
BASE_MAESTROV2_CACHE_PATH = os.path.join(BASE_CHCHE_PATH, "maestro-v2.0.0")
BASE_MAESTROV1_CACHE_PATH = os.path.join(BASE_CHCHE_PATH, "maestro-v1.0.0")

SPLIT_NAME = "split"
SPLIT_PEDAL_NAME = "split_pedal"
SPLIT_PEDAL_NOTE_NAME = "split_pedal_note"
BASE_SPLIT_PATH = os.path.join(WORKSPACE, SPLIT_NAME)
BASE_PEDAL_SPLIT_PATH = os.path.join(WORKSPACE, SPLIT_PEDAL_NAME)
BASE_PEDAL_NOTE_SPLIT_PATH = os.path.join(WORKSPACE, SPLIT_PEDAL_NOTE_NAME)

BASE_MAESTROV3_PEDAL_NOTE_SPLIT_PATH = os.path.join(BASE_PEDAL_NOTE_SPLIT_PATH, "maestro-v3.0.0")
BASE_MAESTROV2_PEDAL_NOTE_SPLIT_PATH = os.path.join(BASE_PEDAL_NOTE_SPLIT_PATH, "maestro-v2.0.0")
BASE_MAESTROV1_PEDAL_NOTE_SPLIT_PATH = os.path.join(BASE_PEDAL_NOTE_SPLIT_PATH, "maestro-v1.0.0")

BASE_MAESTROV3_PEDAL_SPLIT_PATH = os.path.join(BASE_PEDAL_SPLIT_PATH, "maestro-v3.0.0")
BASE_MAESTROV2_PEDAL_SPLIT_PATH = os.path.join(BASE_PEDAL_SPLIT_PATH, "maestro-v2.0.0")
BASE_MAESTROV1_PEDAL_SPLIT_PATH = os.path.join(BASE_PEDAL_SPLIT_PATH, "maestro-v1.0.0")

BASE_MAESTROV3_SPLIT_PATH = os.path.join(BASE_SPLIT_PATH, "maestro-v3.0.0")
BASE_MAESTROV2_SPLIT_PATH = os.path.join(BASE_SPLIT_PATH, "maestro-v2.0.0")
BASE_MAESTROV1_SPLIT_PATH = os.path.join(BASE_SPLIT_PATH, "maestro-v1.0.0")
# ====================to ensure the path exist======================
if not os.path.exists(BASE_CHCHE_PATH):
    os.mkdir(BASE_CHCHE_PATH)
if not os.path.exists(BASE_MAESTROV1_CACHE_PATH):
    os.mkdir(BASE_MAESTROV1_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV2_CACHE_PATH):
    os.mkdir(BASE_MAESTROV2_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV3_CACHE_PATH):
    os.mkdir(BASE_MAESTROV3_CACHE_PATH)
if not os.path.exists(BASE_SPLIT_PATH):
    os.mkdir(BASE_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV1_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV1_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV2_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV2_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV3_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV3_SPLIT_PATH)

if not os.path.exists(BASE_PEDAL_CACHE_PATH):
    os.mkdir(BASE_PEDAL_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV1_PEDAL_CACHE_PATH):
    os.mkdir(BASE_MAESTROV1_PEDAL_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV2_PEDAL_CACHE_PATH):
    os.mkdir(BASE_MAESTROV2_PEDAL_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV3_PEDAL_CACHE_PATH):
    os.mkdir(BASE_MAESTROV3_PEDAL_CACHE_PATH)
if not os.path.exists(BASE_PEDAL_SPLIT_PATH):
    os.mkdir(BASE_PEDAL_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV1_PEDAL_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV1_PEDAL_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV2_PEDAL_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV2_PEDAL_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV3_PEDAL_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV3_PEDAL_SPLIT_PATH)

if not os.path.exists(BASE_PEDAL_NOTE_CACHE_PATH):
    os.mkdir(BASE_PEDAL_NOTE_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV1_PEDAL_NOTE_CACHE_PATH):
    os.mkdir(BASE_MAESTROV1_PEDAL_NOTE_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV2_PEDAL_NOTE_CACHE_PATH):
    os.mkdir(BASE_MAESTROV2_PEDAL_NOTE_CACHE_PATH)
if not os.path.exists(BASE_MAESTROV3_PEDAL_NOTE_CACHE_PATH):
    os.mkdir(BASE_MAESTROV3_PEDAL_NOTE_CACHE_PATH)
if not os.path.exists(BASE_PEDAL_NOTE_SPLIT_PATH):
    os.mkdir(BASE_PEDAL_NOTE_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV1_PEDAL_NOTE_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV1_PEDAL_NOTE_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV2_PEDAL_NOTE_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV2_PEDAL_NOTE_SPLIT_PATH)
if not os.path.exists(BASE_MAESTROV3_PEDAL_NOTE_SPLIT_PATH):
    os.mkdir(BASE_MAESTROV3_PEDAL_NOTE_SPLIT_PATH)

# ===================the precison of note===============================
MIN_NOTE_DURATION = 0.01

# ===================you know the constant meaning======================
SAMPLE_RATE = 16000

DEFAULT_VELOCITY = 100
DEFAULT_NOTE_DURATION = 0.01

# ==================spectrograms config=================================
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512
FFT_SIZE = 2048
MEL_LO_HZ = 20.0

# ===================vocabulary config=================================
STEPS_PER_SECOND = 100
MAX_SHIFT_SECONDS = 10
NUM_VELOCITY_BINS = 127

DECODED_EOS_ID = -1
DECODED_INVALID_ID = -2

# =============================optimizer================================
LEARNING_RATE = 0.0005
BETAS = [0.9, 0.98]
EPS = 1e-08
WEIGHT_DECAY = 0.01

# =============================note_model================================
NOTE_MODEL_EPOCHS = 400
NOTE_MODEL_SCHEDULER_WARMUP_RATIO = 1 / (3 * NOTE_MODEL_EPOCHS)

MUSICFORMER_MODEL_EPOCHS = 400
MUSICFORMER_MODEL_SCHEDULER_WARMUP_RATIO = 1 / (3 * NOTE_MODEL_EPOCHS)


PEDAL_MODEL_EPOCHS = 400
PEDAL_MODEL_SCHEDULER_WARMUP_RATIO = 1 / (3 * NOTE_MODEL_EPOCHS)

NOTE_PEDAL_MODEL_EPOCHS = 800
NOTE_PEDAL_MODEL_SCHEDULER_WARMUP_RATIO = 1 / (3 * NOTE_MODEL_EPOCHS)

_SUSTAIN_ON = 1
_SUSTAIN_OFF = 0
