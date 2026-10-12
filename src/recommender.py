"""Recommender engine implementing content-based movie recommendations."""

import difflib
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix

from src.config import DEFAULT_FUZZY_CUTOFF, DEFAULT_TOP_K
from src.data_loader import load_dataset
from src.preprocessor import build_tfidf_matrix, combine_features

logger = logging.getLogger(__name__)


def _clean_str(val: Any) -> str:
    """Helper to safely format string fields and suppress 'nan'."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


class MovieRecommender:
    """Content-based Movie Recommendation Engine using TF-IDF and Cosine Similarity."""

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        include_overview: bool = True,
        stop_words: Optional[str] = "english",
    ):
        """Initialize the recommender.

        Args:
            df: Optional preloaded DataFrame. If None, loaded via load_dataset().
            include_overview: Whether to include movie plot overview in features.
            stop_words: Stop words for TF-IDF vectorizer.
        """
        self.include_overview = include_overview
        self.stop_words = stop_words
        self.df = df if df is not None else load_dataset()
        self.vectorizer = None
        self.tfidf_matrix: Optional[spmatrix] = None
        self._title_lookup: Dict[str, Tuple[str, int]] = {}
        self._fit()

    def _fit(self) -> None:
        """Internal fit method: processes features and builds TF-IDF matrix."""
        # Ensure clean 0-indexed dataframe
        self.df = self.df.reset_index(drop=True)

        # Build normalized title lookup mapping {lowercase_title: (original_title, row_index)}
        self._title_lookup = {}
        for idx, title in enumerate(self.df["Movie_Title"]):
            clean_title = str(title).strip()
            self._title_lookup[clean_title.lower()] = (clean_title, idx)

        # Build combined features & TF-IDF matrix
        combined_text = combine_features(self.df, include_overview=self.include_overview)
        self.vectorizer, self.tfidf_matrix = build_tfidf_matrix(
            combined_text, stop_words=self.stop_words
        )
        logger.info(
            f"Fitted TF-IDF matrix with shape {self.tfidf_matrix.shape} across {len(self.df)} movies."
        )

    def find_close_match(
        self, query: str, cutoff: float = DEFAULT_FUZZY_CUTOFF
    ) -> Optional[Tuple[str, float, int]]:
        """Find the closest matching movie title in the dataset.

        Args:
            query: User's movie query string.
            cutoff: Minimum similarity cutoff for fuzzy matching (0.0 to 1.0).

        Returns:
            Optional[Tuple[str, float, int]]: (matched_title, match_confidence, row_index)
                                              or None if no match is found.
        """
        if not query or not query.strip():
            return None

        clean_query = query.strip()
        lower_query = clean_query.lower()

        # 1. Exact match (case-insensitive)
        if lower_query in self._title_lookup:
            orig_title, row_idx = self._title_lookup[lower_query]
            return orig_title, 1.0, row_idx

        # 2. Substring match (e.g. "avengers" matches "The Avengers")
        substring_matches = [
            (orig, idx)
            for low, (orig, idx) in self._title_lookup.items()
            if lower_query in low
        ]
        if substring_matches:
            # Pick the shortest matching title (closest length)
            best_sub = min(substring_matches, key=lambda item: len(item[0]))
            ratio = difflib.SequenceMatcher(None, lower_query, best_sub[0].lower()).ratio()
            return best_sub[0], max(ratio, 0.75), best_sub[1]

        # 3. Fuzzy matching via difflib on all normalized titles
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
        movie_title: str,
        top_k: int = DEFAULT_TOP_K,
        exclude_self: bool = True,
        cutoff: float = DEFAULT_FUZZY_CUTOFF,
    ) -> Dict[str, Any]:
        """Generate top-K movie recommendations for a given title.

        Args:
            movie_title: Target movie name.
            top_k: Number of recommendations to return.
            exclude_self: Whether to filter out the query movie from recommendations.
            cutoff: Fuzzy match confidence cutoff.

        Returns:
            Dict[str, Any]: Recommendation result dictionary containing:
                - 'query': original query string
                - 'matched_title': closest dataset title (or None)
                - 'match_confidence': fuzzy match confidence score (or 0.0)
                - 'recommendations': list of recommendation dictionaries
                - 'is_fallback': boolean indicating if popular fallbacks were returned
        """
        match_result = self.find_close_match(movie_title, cutoff=cutoff)

        if match_result is None:
            # Fallback to popular movies if title not recognized
            popular_df = self.get_popular_movies(top_k=top_k)
            return {
                "query": movie_title,
                "matched_title": None,
                "match_confidence": 0.0,
                "recommendations": popular_df.to_dict(orient="records"),
                "is_fallback": True,
                "message": f"No close match found for '{movie_title}'. Showing popular movies instead.",
            }

        matched_title, confidence, row_idx = match_result

        # Compute cosine similarity for the target row vector against all movies
        query_vector = self.tfidf_matrix[row_idx]
        sim_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Enumerate and sort by score descending
        ranked_indices = np.argsort(sim_scores)[::-1]

        recommendations = []
        for idx in ranked_indices:
            if exclude_self and idx == row_idx:
                continue

            movie_row = self.df.iloc[idx]
            recommendations.append(
                {
                    "rank": len(recommendations) + 1,
                    "movie_id": int(movie_row.get("Movie_ID", idx)),
                    "title": _clean_str(movie_row.get("Movie_Title", "")),
                    "similarity_score": round(float(sim_scores[idx]), 4),
                    "match_percentage": f"{round(float(sim_scores[idx]) * 100, 1)}%",
                    "genre": _clean_str(movie_row.get("Movie_Genre", "")),
                    "director": _clean_str(movie_row.get("Movie_Director", "")),
                    "cast": _clean_str(movie_row.get("Movie_Cast", "")),
                    "tagline": _clean_str(movie_row.get("Movie_Tagline", "")),
                    "overview": _clean_str(movie_row.get("Movie_Overview", "")),
                    "vote_average": float(movie_row.get("Movie_Vote", 0.0) or 0.0),
                    "vote_count": int(movie_row.get("Movie_Vote_Count", 0) or 0),
                    "release_date": _clean_str(movie_row.get("Movie_Release_Date", "")),
                    "runtime": int(float(movie_row.get("Movie_Runtime", 0.0) or 0.0)),
                    "popularity": round(float(movie_row.get("Movie_Popularity", 0.0) or 0.0), 2),
                    "homepage": _clean_str(movie_row.get("Movie_Homepage", "")),
                }
            )

            if len(recommendations) >= top_k:
                break

        return {
            "query": movie_title,
            "matched_title": matched_title,
            "match_confidence": round(confidence, 3),
            "recommendations": recommendations,
            "is_fallback": False,
            "message": f"Top {len(recommendations)} recommendations based on '{matched_title}'.",
        }

    def get_popular_movies(self, top_k: int = DEFAULT_TOP_K) -> pd.DataFrame:
        """Returns top popular movies from the dataset based on popularity and votes.

        Args:
            top_k: Number of popular movies to return.

        Returns:
            pd.DataFrame: Top popular movies with relevant metadata.
        """
        df_copy = self.df.copy()

        # Determine best sorting column
        if "Movie_Popularity" in df_copy.columns:
            df_copy["_sort_score"] = pd.to_numeric(
                df_copy["Movie_Popularity"], errors="coerce"
            ).fillna(0)
        elif "Movie_Vote_Count" in df_copy.columns:
            df_copy["_sort_score"] = pd.to_numeric(
                df_copy["Movie_Vote_Count"], errors="coerce"
            ).fillna(0)
        else:
            df_copy["_sort_score"] = 0

        popular = df_copy.sort_values(by="_sort_score", ascending=False).head(top_k)

        results = []
        for rank, (_, row) in enumerate(popular.iterrows(), 1):
            results.append(
                {
                    "rank": rank,
                    "movie_id": int(row.get("Movie_ID", 0)),
                    "title": _clean_str(row.get("Movie_Title", "")),
                    "similarity_score": 0.0,
                    "match_percentage": "Popular",
                    "genre": _clean_str(row.get("Movie_Genre", "")),
                    "director": _clean_str(row.get("Movie_Director", "")),
                    "cast": _clean_str(row.get("Movie_Cast", "")),
                    "tagline": _clean_str(row.get("Movie_Tagline", "")),
                    "overview": _clean_str(row.get("Movie_Overview", "")),
                    "vote_average": float(row.get("Movie_Vote", 0.0) or 0.0),
                    "vote_count": int(row.get("Movie_Vote_Count", 0) or 0),
                    "release_date": _clean_str(row.get("Movie_Release_Date", "")),
                    "runtime": int(float(row.get("Movie_Runtime", 0.0) or 0.0)),
                    "popularity": round(float(row.get("Movie_Popularity", 0.0) or 0.0), 2),
                    "homepage": _clean_str(row.get("Movie_Homepage", "")),
                }
            )

        return pd.DataFrame(results)

    def get_all_titles(self) -> List[str]:
        """Returns a sorted list of all unique movie titles for autocomplete/dropdowns."""
        return sorted(self.df["Movie_Title"].dropna().unique().tolist())

    def get_genres_list(self) -> List[str]:
        """Returns sorted list of common genres found in the dataset."""
        return [
            "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
            "Drama", "Family", "Fantasy", "History", "Horror", "Music",
            "Mystery", "Romance", "Science Fiction", "Thriller", "War", "Western"
        ]
