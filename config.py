# ---- Config -- extended 1993-2024 dataset, 31 years, ~19 crises, international indices ----
#
# Saved as config.py (not config-4.py): features.py does `import config`,
# which resolves only under this exact filename.
#
# Only WINDOWS_HURST and TICKERS_SECTOR are read by the current pipeline
# (features.py). Everything else here is either historical (superseded by
# the inline CFG class in main_v304_soft_labels.py) or was already dead in
# the uploaded file: LAMBDA_CONF and LAMBDA_MARGIN were each defined twice
# with different values (0.5/0.3 further down, then 0.3/0.2 in the V24
# block); Python silently keeps the last one, so only that value ever had
# any effect. This version keeps a single definition of each -- same
# runtime behavior, one less trap for whoever reads this next.
import torch

# ---- Original assets ----
TICKERS_MARKET = ["SPY", "QQQ", "IWM", "DIA", "VTI"]
TICKERS_SECTOR = ["XLK", "XLF", "XLE", "XLV", "XLI",
                   "XLU", "XLB", "XLC", "XLY", "XLP", "XLRE"]
TICKERS_MACRO = ["TLT", "SHY", "IEF", "HYG", "LQD",
                  "GLD", "TIP", "UUP", "USO", "DBC"]
TICKERS_INTL = ["EFA", "EEM", "FXI", "EWJ", "VGK"]

# ---- Diversified assets (V21) ----
TICKERS_INTL_NEW = ["EWZ", "EWY", "EWT", "EWA", "EWG"]
TICKERS_CMDTY = ["SLV", "DBA", "GDX", "EZU"]
TICKERS_BOND_NEW = ["AGG", "EMB", "JNK"]
TICKERS_SECT_NEW = ["IBB", "KRE", "XHB", "IYT", "XRT"]
TICKERS_FX = ["FXE", "FXY", "FXB"]

# ---- International indices (long history from 1990+) ----
# Carry the memory of pre-2007 crises: 1994 (Fed rate hikes), 1997 (Asia),
# 1998 (LTCM), 2000-2002 (dot-com), 2001 (9/11).
TICKERS_INDICES = ["^GSPC", "^IXIC", "^N225", "^FTSE",
                    "^GDAXI", "^STOXX50E", "^FCHI", "^HSI"]

# ---- Fear gauge and long-history commodities ----
TICKERS_FEAR = ["^VIX"]              # since 1990
TICKERS_COMMO_LH = ["GC=F", "CL=F"]  # gold and oil, since 1975+

ALL_TICKERS = (TICKERS_MARKET + TICKERS_SECTOR + TICKERS_MACRO +
               TICKERS_INTL + TICKERS_INTL_NEW + TICKERS_CMDTY +
               TICKERS_BOND_NEW + TICKERS_SECT_NEW + TICKERS_FX +
               TICKERS_INDICES + TICKERS_FEAR + TICKERS_COMMO_LH)

TARGET = "SPY"
START_DATE = "1993-02-01"  # SPY inception is 1993-01-29
END_DATE = "2024-01-01"

WINDOWS_ALL = [5, 10, 15, 20, 30, 40, 50, 60, 90, 120, 180, 252]
WINDOWS_HURST = [60, 90, 120, 180, 252]

# Feature split (204 total)
SLOW_IDX = list(range(0, 40)) + list(range(160, 200))
FAST_IDX = list(range(40, 160)) + list(range(200, 204))
SLOW_DIM = 80
FAST_DIM = 124

# Architecture
LATENT_DIM = 4
N_VERLET_STEPS = 6
DT = 0.1
POTENTIAL_DIM = 32
FORCE_DIM = 32
DROPOUT = 0.3
NOISE_STD = 0.05

N_REGIMES = 2
CLASS_WEIGHTS = {0: 1.0, 1: 2.0}
REGIME_NAMES = {0: "Stable", 1: "Critical", 2: "Transition", 3: "OOD"}

LAMBDA_MARGIN = 0.2
LAMBDA_ENERGY = 0.1
LAMBDA_TREND = 0.0  # disabled (unstable)

FRICTION_GAMMA = 0.3
THERMAL_TEMP = 0.05
ENSEMBLE_SEEDS = [42, 123, 456]
ASYM_FRICTION_ALPHA = 0.0  # disabled (unstable)
DECISION_THRESHOLD = 0.50

BATCH_SIZE = 64
LR = 3e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# V24 -- drawdown regression (replaces binary classification)
DD_HORIZON = 20        # forward days
DD_HIGH = 0.03          # "crisis" threshold for evaluation (3%)
QUANTILE_TAU = 0.75     # conservatism: penalizes under-estimation 3x
LAMBDA_CONF = 0.3
