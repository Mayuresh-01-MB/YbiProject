"""Data loader module for the Movie Recommendation System."""

import logging
from pathlib import Path
from typing import Optional, Union
import pandas as pd

from src.config import DEFAULT_DATASET_PATH, REMOTE_DATASET_URL

logger = logging.getLogger(__name__)


def load_dataset(
    path_or_url: Optional[Union[str, Path]] = None,
    use_remote_fallback: bool = True,
) -> pd.DataFrame:
    """Load the movies dataset from local cache or remote URL.

    Args:
        path_or_url: Optional local path or URL to the CSV file.
                     If None, tries DEFAULT_DATASET_PATH first.
        use_remote_fallback: If True, falls back to REMOTE_DATASET_URL when
                             the local file is not found.

    Returns:
        pd.DataFrame: Cleaned movies dataframe with contiguous 0-based integer index.
    """
    df: Optional[pd.DataFrame] = None
    target_path = Path(path_or_url) if path_or_url else DEFAULT_DATASET_PATH

    # Attempt to load from target path (local file)
    if isinstance(target_path, Path) and target_path.exists():
        logger.info(f"Loading dataset from local path: {target_path}")
        df = pd.read_csv(target_path)
    elif isinstance(path_or_url, str) and (
        path_or_url.startswith("http://") or path_or_url.startswith("https://")
    ):
        logger.info(f"Loading dataset from provided URL: {path_or_url}")
        df = pd.read_csv(path_or_url)
    elif use_remote_fallback:
        logger.info(
            f"Local file not found at {target_path}. Falling back to remote URL: {REMOTE_DATASET_URL}"
        )
        df = pd.read_csv(REMOTE_DATASET_URL)
    else:
        raise FileNotFoundError(f"Dataset not found at {target_path}")

    # Ensure contiguous 0-based indexing
    df = df.reset_index(drop=True)

    # Clean Movie_Title whitespace
    if "Movie_Title" in df.columns:
        df["Movie_Title"] = df["Movie_Title"].astype(str).str.strip()

    return df

