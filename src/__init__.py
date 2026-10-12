"""YBI Foundation Movie & TV Series Recommendation Package."""

from src.data_loader import load_dataset
from src.recommender import MovieRecommender
from src.series_recommender import SeriesRecommender, load_series_dataset

__all__ = [
    "load_dataset",
    "MovieRecommender",
    "load_series_dataset",
    "SeriesRecommender",
]
__version__ = "2.0.0"
