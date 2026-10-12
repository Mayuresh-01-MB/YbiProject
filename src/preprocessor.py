"""Preprocessor module for feature combination and TF-IDF vectorization."""

from typing import List, Optional, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

from src.config import TEXT_FEATURE_COLUMNS, DEFAULT_STOP_WORDS


def combine_features(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    include_overview: bool = True,
) -> pd.Series:
    """Combines textual movie metadata into a unified text feature series.

    Args:
        df: Input DataFrame containing movie features.
        feature_columns: List of columns to combine. Defaults to TEXT_FEATURE_COLUMNS.
        include_overview: Whether to include 'Movie_Overview' in the feature text.

    Returns:
        pd.Series: Combined text representations for each movie.
    """
    cols = list(feature_columns or TEXT_FEATURE_COLUMNS)
    if include_overview and "Movie_Overview" in df.columns and "Movie_Overview" not in cols:
        cols.append("Movie_Overview")

    # Filter columns that actually exist in the dataframe
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        raise ValueError("None of the specified feature columns exist in the DataFrame.")

    # Fill NaN and join row-wise
    combined = df[valid_cols].fillna("").astype(str).agg(" ".join, axis=1)
    return combined.str.strip()


def build_tfidf_matrix(
    text_series: pd.Series,
    stop_words: Optional[str] = DEFAULT_STOP_WORDS,
    max_features: Optional[int] = None,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, spmatrix]:
    """Fits TfidfVectorizer on the combined text features and returns the matrix.

    Args:
        text_series: Series of combined text features.
        stop_words: Stop words parameter for TfidfVectorizer ('english' or None).
        max_features: Optional maximum number of vocabulary features.
        ngram_range: Range of n-grams (default unigram + bigram (1, 2)).

    Returns:
        Tuple[TfidfVectorizer, spmatrix]: Fitted vectorizer and sparse TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer(
        stop_words=stop_words,
        max_features=max_features,
        ngram_range=ngram_range,
    )
    tfidf_matrix = vectorizer.fit_transform(text_series)
    return vectorizer, tfidf_matrix

