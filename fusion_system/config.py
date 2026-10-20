"""
config.py — single source of truth for all hyperparameters and paths.
Every other module imports from here; change once, applies everywhere.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths  (adjust to your machine layout)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent

DATA_ROOT        = ROOT.parent / "dataset"
PARQUET_PATH     = DATA_ROOT / "dataset_V2.parquet"     # dataset/dataset_V2.parquet

THERMAL_CSV_DIR  = DATA_ROOT / "Thermographie" / "Thermographie"
THERMAL_PNG_DIR  = ROOT / "thermal_pngs"                # Ashwin's pre-converted PNGs (optional)

CV_IMAGE_DIR     = DATA_ROOT / "CV_images_Vorversuche_PP" / "Rohbilder"
CV_XML_DIR       = CV_IMAGE_DIR                          # XMLs live in same folder

ARTIFACTS_DIR    = ROOT / "artifacts"
MODELS_DIR       = ROOT / "saved_models"
PLOTS_DIR        = ROOT / "plots"
LOGS_DIR         = ROOT / "logs"

for _d in [ARTIFACTS_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset columns
# ---------------------------------------------------------------------------
LABEL_COLS = [
    "LBL_SinkMarks", "LBL_SprueCircle", "LBL_Underfilled",
    "LBL_OldGranulate", "LBL_StreaksLevel1", "LBL_StreaksLevel2",
    "LBL_StreaksLevel3", "LBL_NOK",
]
N_LABELS = len(LABEL_COLS)

PRIMARY_TARGET = "LBL_NOK"                              # binary headline metric

TABULAR_PREFIXES = ("SET_", "QUA_", "ENV_", "CALC_", "DOS_", "DRY_", "SIM_")
SEQUENCE_PREFIX  = "DXP_"

GROUP_COL    = "MET_ExperimentNumber"                   # group-aware split key
CYCLE_ID_COL = "MET_MachineCycleID"
JOB_CYCLE_COL= "MET_JobCycleID"
MATERIAL_COL = "MET_MaterialName"

IR_IMG1_COL  = "IR_Image1Name"
IR_IMG2_COL  = "IR_Image2Name"
CV_IMG1_COL  = "CV_Image1Name"
CV_IMG2_COL  = "CV_Image2Name"
CV_IMG3_COL  = "CV_Image3Name"

# ROI temperature features extracted by SKZ (used in physics head)
THERMAL_ROI_COLS = [
    "IR_Img1TempSprue", "IR_Img1TempDome",
    "IR_Img1TempEdgeHor", "IR_Img1TempEdgeVer", "IR_Img1TempFull",
    "IR_Img2TempSprue", "IR_Img2TempDome",
    "IR_Img2TempEdgeHor", "IR_Img2TempEdgeVer", "IR_Img2TempFull",
]

WARPAGE_COLS = ["CV_Width1", "CV_Width2", "CV_Width3", "CV_Warpage"]

# ---------------------------------------------------------------------------
# Modality switches  (disable any branch you don't have data for)
# ---------------------------------------------------------------------------
USE_THERMAL  = True
USE_VISUAL   = True
USE_SEQUENCE = True
USE_TABULAR  = True

# ---------------------------------------------------------------------------
# Image sizes
# ---------------------------------------------------------------------------
THERMAL_H, THERMAL_W  = 480, 640       # raw CSV resolution
THERMAL_INPUT_SIZE    = 224            # resize target for EfficientNet

CV_ORIG_H, CV_ORIG_W  = 2048, 2448    # raw BMP resolution
CV_INPUT_SIZE         = 224            # resize target per section
VISUAL_RESIZE         = CV_INPUT_SIZE  # alias used in analysis/explainability

# ---------------------------------------------------------------------------
# Embedding dimensions
# ---------------------------------------------------------------------------
THERMAL_EMB_DIM  = 512    # EfficientNet-B0 backbone output → Linear(1280→512)
VISUAL_EMB_DIM   = 512    # ResNet-50 → Linear(2048→512)
TCN_EMB_DIM      = 256    # TCN pooled output
TABULAR_EMB_DIM  = 192    # MLP output

FUSION_TOKEN_DIM = 256    # all modalities projected to this before fusion
FUSION_N_HEADS   = 4
FUSION_N_LAYERS  = 2
FUSION_DROPOUT   = 0.1

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEED          = 42
BATCH_SIZE    = 16
NUM_EPOCHS    = 60
LR_BACKBONE   = 1e-5     # pretrained backbone
LR_HEAD       = 1e-4     # new heads
WEIGHT_DECAY  = 1e-4
PATIENCE      = 10       # early stopping on val macro-F1
GRAD_CLIP     = 1.0

# DANN (domain-adversarial) loss weight schedule
DANN_LAMBDA_MAX = 0.5    # maximum gradient reversal weight
DANN_WARMUP_EP  = 10     # epochs before DANN kicks in

# Focal loss params (for sequence / main head)
FOCAL_ALPHA   = 0.75
FOCAL_GAMMA   = 2.0

# ---------------------------------------------------------------------------
# Sequence (TCN)
# ---------------------------------------------------------------------------
TCN_TARGET_LEN   = 4096          # resample all DXP sequences to this length
TCN_IN_CHANNELS  = 8             # key DXP channels (pressure, position, temp)
TCN_CHANNEL_LIST = [64, 128, 256]
TCN_KERNEL_SIZE  = 3
TCN_DILATIONS    = [1, 2, 4]
TCN_DROPOUT      = 0.2

KEY_DXP_CHANNELS = [
    "DXP_Inj1PrsAct",     # injection pressure
    "DXP_Inj1PosAct",     # screw position
    "DXP_Inj1VelAct",     # injection velocity
    "DXP_HldPrsAct",      # holding pressure
    "DXP_ClpFrcAct",      # clamping force
    "DXP_MldTmpEjt",      # mould temp ejector side
    "DXP_MldTmpNzl",      # mould temp nozzle side
    "DXP_DosVlmAct",      # dosing volume
]

# ---------------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------------
N_GROUP_FOLDS    = 5       # group k-fold folds
VAL_FRACTION     = 0.15    # of training data for early stopping
TEST_FRACTION    = 0.15    # held-out test set (time-aware: last 15% by cycle ID)

# Threshold search grid for per-label tuning
THRESHOLD_GRID   = [round(x * 0.05, 2) for x in range(2, 19)]  # 0.10 … 0.90

# ---------------------------------------------------------------------------
# Drift monitor
# ---------------------------------------------------------------------------
DRIFT_PSI_THRESHOLD  = 0.2   # PSI > 0.2 → significant shift
DRIFT_WINDOW_SIZE    = 20    # number of recent cycles to compare vs. reference

# ---------------------------------------------------------------------------
# Counterfactual recommender
# ---------------------------------------------------------------------------
CF_MAX_CANDIDATES    = 5     # top-K setpoint suggestions to return
CF_SET_COLS = [
    "SET_CylinderTemperature", "SET_ToolTemperature",
    "SET_InjectionSpeed",      "SET_HoldingPressure1",
    "SET_HoldingPressure2",    "SET_CoolingTime",
]
