"""Configuration settings for the Movie Recommendation System."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DEFAULT_DATASET_PATH = DATASET_DIR / "Movies_Recommendation.csv"

# Remote dataset URL fallback
REMOTE_DATASET_URL = (
    "https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv"
)

# Feature columns used for content-based similarity
TEXT_FEATURE_COLUMNS = [
    "Movie_Genre",
    "Movie_Keywords",
    "Movie_Tagline",
    "Movie_Cast",
    "Movie_Director",
]

# Additional metadata columns preserved for recommendation display
METADATA_COLUMNS = [
    "Movie_ID",
    "Movie_Title",
    "Movie_Genre",
    "Movie_Director",
    "Movie_Cast",
    "Movie_Tagline",
    "Movie_Overview",
    "Movie_Vote",
    "Movie_Vote_Count",
    "Movie_Popularity",
    "Movie_Release_Date",
]

# Recommender defaults
DEFAULT_TOP_K = 10
DEFAULT_FUZZY_CUTOFF = 0.6
DEFAULT_STOP_WORDS = "english"

# Series Paths & Columns
DEFAULT_SERIES_DATASET_PATH = DATASET_DIR / "TV_Shows_Recommendation.csv"
REMOTE_SERIES_DATASET_URL = "https://raw.githubusercontent.com/ankoorb/IMDB/master/tv_shows.csv"
SERIES_TEXT_FEATURE_COLUMNS = ["Series_Genre", "Series_Overview"]
