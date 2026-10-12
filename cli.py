"""Command Line Interface (CLI) for Movie Recommendation System."""

import argparse
import sys
from src.recommender import MovieRecommender

# Ensure UTF-8 output on Windows consoles if supported
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Content-Based Movie Recommendation System (YBI Project)"
    )
    parser.add_argument(
        "-m",
        "--movie",
        type=str,
        help="Title of your favourite movie to get recommendations for.",
    )
    parser.add_argument(
        "-k",
        "--top",
        type=int,
        default=10,
        help="Number of movie recommendations to return (default: 10).",
    )
    parser.add_argument(
        "--popular",
        action="store_true",
        help="Display the top popular movies in the dataset.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  YBI Foundation - Movie Recommendation System")
    print("=" * 60)
    print("Initializing recommender engine...")
    recommender = MovieRecommender()

    if args.popular:
        print(f"\nTop {args.top} Popular Movies:")
        popular_df = recommender.get_popular_movies(top_k=args.top)
        for _, row in popular_df.iterrows():
            print(
                f" {row['rank']:2d}. {row['title']} | Genre: {row['genre']} | Rating: {row['vote_average']}/10"
            )
        return

    movie_query = args.movie
    if not movie_query:
        movie_query = input("\nEnter your favourite movie name: ").strip()

    if not movie_query:
        print("No movie name provided. Exiting.")
        sys.exit(0)

    result = recommender.recommend(movie_query, top_k=args.top)

    if result["is_fallback"]:
        print(f"\n[Notice] {result['message']}\n")
    else:
        print(
            f"\n[Matched] '{result['matched_title']}' (Confidence: {result['match_confidence'] * 100:.1f}%)"
        )
        print(f"Showing Top {len(result['recommendations'])} Recommendations:\n")

    for rec in result["recommendations"]:
        score_str = (
            f"Similarity: {rec['match_percentage']}"
            if not result["is_fallback"]
            else "Popular"
        )
        print(f"  {rec['rank']:2d}. {rec['title']}")
        print(f"      {score_str} | Genre: {rec['genre']} | Director: {rec['director']}")
        if rec.get("tagline"):
            print(f"      Tagline: \"{rec['tagline']}\"")
        print()

    print("=" * 60)


if __name__ == "__main__":
    main()

