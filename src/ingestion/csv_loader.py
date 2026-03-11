import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_csv(file_path: Path | str) -> pd.DataFrame:
    """
    Load tyre dataset from CSV.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"CSV file not found at {path}")
        raise FileNotFoundError(f"CSV file not found at {path}")
    
    logger.info(f"Loading CSV data from {path}")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} rows.")
        # Replace empty strings and unhelpful values
        df = df.fillna("NA")
        df = df.replace({"#N/A": "None", " ": "NA", "": "NA"})
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise e
