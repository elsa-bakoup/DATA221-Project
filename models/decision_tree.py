import pandas as pd

from pathlib import Path
import sys
from shared.paths import DATA_SPLIT

# Make sure the root is linked
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_split_data():
    X_train = pd.read_csv(DATA_SPLIT / "X_train.csv")
    X_test = pd.read_csv(DATA_SPLIT / "X_test.csv")
    y_train = pd.read_csv(DATA_SPLIT / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_SPLIT / "y_test.csv").squeeze("columns")

    return X_train, X_test, y_train, y_test