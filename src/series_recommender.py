"""Recommender engine specifically for TV Series & Shows."""

import difflib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    DEFAULT_SERIES_DATASET_PATH,
    REMOTE_SERIES_DATASET_URL,
    DEFAULT_FUZZY_CUTOFF,
    DEFAULT_TOP_K,
)

logger = logging.getLogger(__name__)


def _clean_str(val: Any) -> str:
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


def load_series_dataset(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load the TV Shows dataset from local cache or remote URL fallback."""
    target_path = Path(path) if path else DEFAULT_SERIES_DATASET_PATH

    if target_path.exists():
        df = pd.read_csv(target_path)
    else:
        logger.info(f"Local series dataset not found. Fetching from remote URL: {REMOTE_SERIES_DATASET_URL}")
        df = pd.read_csv(REMOTE_SERIES_DATASET_URL)
        df["Series_ID"] = range(1, len(df) + 1)
        df["Series_Title"] = df["title"].astype(str).str.strip()
        df["Series_Genre"] = df["genre"].fillna("").astype(str)
        df["Series_Year"] = df["year"].fillna("").astype(str)
        df["Series_Runtime"] = df["runtime"].fillna("").astype(str)
        df["Series_Rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0)
        df["Series_Votes"] = pd.to_numeric(df["votes"].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
        df["Series_Overview"] = df["text"].fillna("").astype(str)

    df = df.reset_index(drop=True)
    df["Series_Title"] = df["Series_Title"].astype(str).str.strip()
    return df


class SeriesRecommender:
    """Content-based TV Series Recommendation Engine."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df if df is not None else load_series_dataset()
        self.vectorizer = None
        self.tfidf_matrix = None
        self._title_lookup: Dict[str, Tuple[str, int]] = {}
        self._fit()

    def _fit(self) -> None:
        self.df = self.df.reset_index(drop=True)

        self._title_lookup = {}
        for idx, title in enumerate(self.df["Series_Title"]):
            clean_title = str(title).strip()
            self._title_lookup[clean_title.lower()] = (clean_title, idx)

        # Combine Genre and Overview features
        combined_text = (
            self.df["Series_Genre"].fillna("").astype(str)
            + " "
            + self.df["Series_Overview"].fillna("").astype(str)
        ).str.strip()

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(combined_text)
        logger.info(f"Fitted Series TF-IDF matrix: {self.tfidf_matrix.shape} across {len(self.df)} shows.")

    def find_close_match(
        self, query: str, cutoff: float = DEFAULT_FUZZY_CUTOFF
    ) -> Optional[Tuple[str, float, int]]:
        if not query or not query.strip():
            return None

        clean_query = query.strip()
        lower_query = clean_query.lower()

        # 1. Exact match
        if lower_query in self._title_lookup:
            orig_title, row_idx = self._title_lookup[lower_query]
            return orig_title, 1.0, row_idx

        # 2. Substring match
        substring_matches = [
            (orig, idx)
            for low, (orig, idx) in self._title_lookup.items()
            if lower_query in low
        ]
        if substring_matches:
            best_sub = min(substring_matches, key=lambda item: len(item[0]))
            ratio = difflib.SequenceMatcher(None, lower_query, best_sub[0].lower()).ratio()
            return best_sub[0], max(ratio, 0.75), best_sub[1]

        # 3. Fuzzy match
        all_lower_titles = list(self._title_lookup.keys())
        matches = difflib.get_close_matches(lower_query, all_lower_titles, n=1, cutoff=cutoff)
        if matches:
            matched_lower = matches[0]
            orig_title, row_idx = self._title_lookup[matched_lower]
            ratio = difflib.SequenceMatcher(None, lower_query, matched_lower).ratio()
            return orig_title, ratio, row_idx

        return None

    def recommend(
        self,
        series_title: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_self: bool = True,
        cutoff: float = DEFAULT_FUZZY_CUTOFF,
    ) -> Dict[str, Any]:
        match_result = self.find_close_match(series_title, cutoff=cutoff)

        if match_result is None:
            popular_df = self.get_popular_series(top_k=top_k)
            return {
                "query": series_title,
                "matched_title": None,
                "match_confidence": 0.0,
                "recommendations": popular_df.to_dict(orient="records"),
                "is_fallback": True,
                "message": f"No close match found for TV Show '{series_title}'. Showing popular series instead.",
            }

        matched_title, confidence, row_idx = match_result
        query_vector = self.tfidf_matrix[row_idx]
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(sim_scores)[::-1]

        recommendations = []
        for idx in ranked_indices:
            if exclude_self and idx == row_idx:
                continue

            row = self.df.iloc[idx]
            recommendations.append(
                {
                    "rank": len(recommendations) + 1,
                    "series_id": int(row.get("Series_ID", idx)),
                    "title": _clean_str(row.get("Series_Title", "")),
                    "similarity_score": round(float(sim_scores[idx]), 4),
                    "match_percentage": f"{round(float(sim_scores[idx]) * 100, 1)}%",
                    "genre": _clean_str(row.get("Series_Genre", "")),
                    "year": _clean_str(row.get("Series_Year", "")),
                    "runtime": _clean_str(row.get("Series_Runtime", "")),
                    "overview": _clean_str(row.get("Series_Overview", "")),
                    "vote_average": float(row.get("Series_Rating", 0.0) or 0.0),
                    "vote_count": int(row.get("Series_Votes", 0) or 0),
                    "type": "TV Series",
                }
            )

            if len(recommendations) >= top_k:
                break

        return {
            "query": series_title,
            "matched_title": matched_title,
            "match_confidence": round(confidence, 3),
            "recommendations": recommendations,
            "is_fallback": False,
            "message": f"Top {len(recommendations)} TV Shows based on '{matched_title}'.",
        }

    def get_popular_series(self, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        df_copy = self.df.copy()
        df_copy["_sort_score"] = pd.to_numeric(df_copy["Series_Votes"], errors="coerce").fillna(0)
        popular = df_copy.sort_values(by="_sort_score", ascending=False).head(top_k)

        results = []
        for rank, (_, row) in enumerate(popular.iterrows(), 1):
            results.append(
                {
                    "rank": rank,
                    "series_id": int(row.get("Series_ID", 0)),
                    "title": _clean_str(row.get("Series_Title", "")),
                    "similarity_score": 0.0,
                    "match_percentage": "Popular",
                    "genre": _clean_str(row.get("Series_Genre", "")),
                    "year": _clean_str(row.get("Series_Year", "")),
                    "runtime": _clean_str(row.get("Series_Runtime", "")),
                    "overview": _clean_str(row.get("Series_Overview", "")),
                    "vote_average": float(row.get("Series_Rating", 0.0) or 0.0),
                    "vote_count": int(row.get("Series_Votes", 0) or 0),
                    "type": "TV Series",
                }
            )

        return pd.DataFrame(results)

    def get_all_titles(self) -> List[str]:
        return sorted(self.df["Series_Title"].dropna().unique().tolist())

    def get_genres_list(self) -> List[str]:
        return [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
            "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi", "Thriller", "War"
        ]

