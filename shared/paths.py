from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_CLEANED = ROOT / "data" / "cleaned"
DATA_SPLIT = ROOT / "data" / "split"

RESULTS = ROOT / "results"
RESULTS_TUNING = RESULTS / "tuning"
RESULTS_METRICS = RESULTS / "metrics"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_INTERPRETABILITY = RESULTS / "interpretability"