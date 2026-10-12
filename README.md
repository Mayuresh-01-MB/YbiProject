# 🎬⚡ CineMatch 3D | Movie & TV Series Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)](https://pytest.org/)

An end-to-end, content-based **Movie & TV Series Recommendation System** developed as part of the **YBI Foundation Data Science & Machine Learning Internship (Problem Statement 5)**.

The system analyzes movie and television series metadata across **6,680+ titles (4,760 Movies + 1,926 TV Shows)**, computes semantic text representations using **TF-IDF Vectorization**, and generates personalized recommendations via **Cosine Similarity** with separate dedicated dimensions for Movies and Series, futuristic 3D cyberpunk graphics, and tactile sorting controls.

---

## 🌟 Key Features

- **Dual Dimension Architecture**: Separate dedicated realms for **🎬 Movies (4,760 films)** and **📺 TV Series (1,926 shows)**.
- **Content-Based Semantic Filtering**: Evaluates *Genres*, *Keywords*, *Directors/Creators*, *Cast*, and *Storyline Synopsis*.
- **High-to-Low / Low-to-High Star Sorting**: Instant dynamic sorting by Star Rating, Best Semantic Match, Popularity/Votes, and Release Year.
- **Futuristic 3D Cyber HUD Interface**: Neon holographic aesthetics, 3D tactile push buttons, glowing border reflections, and dark-glassmorphic styling.
- **Smart Toggle Switches**: Real-time neural calibrations for Semantic Plot Matching, Anti-Self Safeguard, and Elite 8.0+ filters.
- **Robust Fuzzy Matching & Fallback**: Case-insensitive fuzzy search handling typos and partial queries with trending fallbacks.
- **Self-Match Exclusion**: Filters out the input title so users receive 100% new recommendations.
- **Interactive Holographic Watchlist**: Bookmark movies and TV shows into a session-persistent HUD watchlist.
- **10 Comprehensive Automated Tests**: Full test coverage across data ingestion, recommendation accuracy, sorting, and edge cases.

---

## 📁 Repository Structure

```text
YbiProject/
│
├── .gitignore                      # Git ignore rules for Python, Jupyter, and caches
├── README.md                       # Comprehensive project documentation
├── requirements.txt                # Production and development dependencies
├── PS-5                            # Problem Statement 5 specification
│
├── dataset/                        # Dataset directory (cached locally)
│   ├── Movies_Recommendation.csv   # 4,760 TMDB Movies dataset
│   └── TV_Shows_Recommendation.csv # 1,926 IMDb TV Series dataset
│
├── src/                            # Modular Recommendation Engines
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Configuration, feature lists, and thresholds
│   ├── data_loader.py              # Local loading with remote GitHub fallback
│   ├── preprocessor.py             # Feature combination & TF-IDF vectorizer
│   ├── recommender.py              # Core MovieRecommender engine
│   └── series_recommender.py       # Core SeriesRecommender engine
│
├── tests/                          # Automated Unit Test Suite
│   ├── __init__.py
│   └── test_recommender.py         # 10 comprehensive unit tests
│
├── app.py                          # Streamlit futuristic 3D web application
├── cli.py                          # Command-line interface for terminal queries
└── YbiFoundation.ipynb             # Bug-fixed, step-by-step Jupyter Notebook
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Mayuresh-01-MB/YbiProject.git
cd YbiProject
```

### 2. Install Dependencies

Ensure Python 3.9+ is installed, then install requirements:

```bash
pip install -r requirements.txt
```

---

### 3. Launch the Web Application

To run the interactive Streamlit dashboard:

```bash
# Recommended (works regardless of Windows PATH settings):
python -m streamlit run app.py

# Or directly:
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

### 4. Run via Command Line (CLI)

Get top 5 recommendations for a movie:

```bash
python cli.py --movie "Avatar" --top 5
```

Explore trending/popular movies:

```bash
python cli.py --popular --top 5
```

Interactive prompt:

```bash
python cli.py
```

---

### 5. Run the Jupyter Notebook

Open and run `YbiFoundation.ipynb` locally or in Google Colab:

```bash
jupyter notebook YbiFoundation.ipynb
```

All indexing bugs, missing methods, and redundant logic have been resolved. The notebook runs seamlessly from top to bottom.

---

### 6. Run Automated Tests

Execute the unit test suite to verify data loading, indexing alignment, fuzzy matching, and fallback handling:

```bash
python -m pytest tests/ -v
```

All 9 unit tests validate:
- Accurate indexing alignment (e.g. Avatar matches Aliens, Guardians of the Galaxy, Star Trek).
- Self-match filtering.
- Case-insensitivity and typo tolerance.
- Graceful fallback for non-existent movies.

---

## 📊 Recommendation Example: Querying "Avatar"

When querying **"Avatar"**, the corrected recommendation engine identifies:

| Rank | Recommended Movie | Genre | Director | Similarity |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Aliens** | Horror, Action, Thriller, Sci-Fi | James Cameron | **7.3%** |
| **2** | **Star Trek Beyond** | Action, Adventure, Sci-Fi | Justin Lin | **7.0%** |
| **3** | **Moonraker** | Action, Adventure, Thriller, Sci-Fi | Lewis Gilbert | **6.5%** |
| **4** | **Guardians of the Galaxy** | Action, Sci-Fi, Adventure | James Gunn | **6.4%** |
| **5** | **Alien** | Horror, Action, Thriller, Sci-Fi | Ridley Scott | **6.2%** |

*(Previous unpatched versions returned unrelated movies like "Freaky Friday" due to a 1-indexed `Movie_ID` vs 0-indexed DataFrame row alignment bug).*

---

## 🛠️ Technologies Used

- **Language**: Python 3
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning & NLP**: Scikit-learn (`TfidfVectorizer`, `cosine_similarity`), SciPy
- **Text Matching**: Python standard `difflib`
- **Web Framework**: Streamlit
- **Testing**: Pytest

---

## 👤 Author & Internship Details

- **Author**: Mayuresh Bhandari
- **GitHub**: [@Mayuresh-01-MB](https://github.com/Mayuresh-01-MB)
- **Organization**: YBI Foundation
- **Domain**: Data Science & Machine Learning
- **Problem Statement**: PS-5 (Content-Based Recommendation System)
