"""Unit tests for the Movie Recommendation System."""

import pytest
import pandas as pd
from src.data_loader import load_dataset
from src.preprocessor import combine_features, build_tfidf_matrix
from src.recommender import MovieRecommender


@pytest.fixture(scope="module")
def recommender_instance():
    """Module-level fixture to initialize recommender once for all tests."""
    return MovieRecommender()


def test_data_loader():
    """Verify that the dataset loads correctly and has expected columns."""
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Movie_Title" in df.columns
    assert "Movie_ID" in df.columns
    assert len(df) > 4000
    # Ensure index is 0-indexed contiguous
    assert df.index[0] == 0
    assert df.index[-1] == len(df) - 1


def test_feature_combination():
    """Verify feature combination creates a non-empty string series."""
    df = load_dataset()
    combined = combine_features(df, include_overview=True)
    assert isinstance(combined, pd.Series)
    assert len(combined) == len(df)
    assert (combined.str.len() > 0).all()


def test_tfidf_matrix_building():
    """Verify TF-IDF matrix has expected shape and sparse properties."""
    df = load_dataset().head(50)
    combined = combine_features(df)
    vectorizer, matrix = build_tfidf_matrix(combined)
    assert matrix.shape[0] == 50
    assert matrix.shape[1] > 100


def test_recommender_initialization(recommender_instance):
    """Verify MovieRecommender fits properly with titles indexed."""
    recommender = recommender_instance
    assert recommender.tfidf_matrix is not None
    assert len(recommender.get_all_titles()) > 4000


def test_avatar_recommendation_alignment(recommender_instance):
    """Verify indexing fix: 'Avatar' must recommend sci-fi/action movies, NOT unrelated movies."""
    recommender = recommender_instance
    result = recommender.recommend("Avatar", top_k=5)

    assert result["matched_title"] == "Avatar"
    assert result["match_confidence"] == 1.0
    assert not result["is_fallback"]

    rec_titles = [r["title"] for r in result["recommendations"]]
    assert len(rec_titles) == 5

    # Self-exclusion check: Avatar should not be in the recommendations
    assert "Avatar" not in rec_titles

    # Verify that sci-fi / action related movies appear
    expected_matches = {"Aliens", "Alien", "Star Trek Beyond", "Guardians of the Galaxy", "Moonraker"}
    found = any(title in expected_matches for title in rec_titles)
    assert found, f"Expected relevant sci-fi movies, got: {rec_titles}"

    # Regression check: 'The Girl on the Train' or 'Freaky Friday' should NOT be in top 5
    assert "The Girl on the Train" not in rec_titles
    assert "Freaky Friday" not in rec_titles


def test_case_insensitivity_and_fuzzy_matching(recommender_instance):
    """Verify matching handles lowercase, uppercase, and minor typos."""
    recommender = recommender_instance

    # Lowercase
    res_lower = recommender.recommend("avatar", top_k=3)
    assert res_lower["matched_title"] == "Avatar"

    # Uppercase
    res_upper = recommender.recommend("AVATAR", top_k=3)
    assert res_upper["matched_title"] == "Avatar"

    # Minor typo (e.g., 'avater')
    res_typo = recommender.recommend("avater", top_k=3)
    assert res_typo["matched_title"] == "Avatar"
    assert res_typo["match_confidence"] > 0.7


def test_unknown_movie_fallback(recommender_instance):
    """Verify that an unknown movie title returns popular fallbacks without raising IndexError."""
    recommender = recommender_instance
    result = recommender.recommend("NonExistentMovie999xyz_Gibberish", top_k=5)

    assert result["is_fallback"] is True
    assert result["matched_title"] is None
    assert len(result["recommendations"]) == 5
    # Popular fallback returns movies with rank 1..5
    assert result["recommendations"][0]["rank"] == 1


def test_empty_query_fallback(recommender_instance):
    """Verify empty string or whitespace queries handle gracefully."""
    recommender = recommender_instance
    res_empty = recommender.recommend("", top_k=5)
    assert res_empty["is_fallback"] is True

    res_spaces = recommender.recommend("   ", top_k=5)
    assert res_spaces["is_fallback"] is True


def test_popular_movies_dataframe(recommender_instance):
    """Verify get_popular_movies returns valid DataFrame."""
    recommender = recommender_instance
    popular = recommender.get_popular_movies(top_k=5)
    assert isinstance(popular, pd.DataFrame)
    assert len(popular) == 5
    assert "title" in popular.columns
    assert "genre" in popular.columns
    assert "vote_average" in popular.columns


def test_series_recommender():
    """Verify TV Series Recommender functions accurately."""
    from src.series_recommender import SeriesRecommender
    sr = SeriesRecommender()
    assert len(sr.get_all_titles()) > 1500

    # Test Breaking Bad recommendation
    res = sr.recommend("Breaking Bad", top_k=5)
    assert res["matched_title"] == "Breaking Bad"
    assert not res["is_fallback"]
    assert len(res["recommendations"]) == 5
    assert "Breaking Bad" not in [r["title"] for r in res["recommendations"]]

    # Test unknown show fallback
    fallback_res = sr.recommend("CompletelyNonExistentShowXYZ999", top_k=3)
    assert fallback_res["is_fallback"] is True
    assert len(fallback_res["recommendations"]) == 3


