"""
Book Recommendation System - Flask Web Application
====================================================
A web-based ML book recommendation system that implements:
  1. Content-Based Filtering (TF-IDF + Cosine Similarity)
  2. Collaborative Filtering (KNN + SVD)

Provides a dashboard to get personalized recommendations and
compare algorithm performance side-by-side.
"""

import os
import json
from flask import Flask, render_template, request, jsonify

from dataset import load_or_generate_data
from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
from evaluation import run_full_comparison

app = Flask(__name__)

# ---- Global state ----
books_df = None
ratings_df = None
cb_model = None
cf_svd_model = None
cf_knn_model = None
comparison_results = None


def initialize():
    """Load data and train all models on startup."""
    global books_df, ratings_df, cb_model, cf_svd_model, cf_knn_model, comparison_results

    print("Loading dataset...")
    books_df, ratings_df = load_or_generate_data()
    print(f"  {len(books_df)} books, {len(ratings_df)} ratings, "
          f"{ratings_df['user_id'].nunique()} users")

    print("\nTraining Content-Based model...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(books_df, ratings_df)
    print(f"  Done in {cb_model.train_time:.3f}s")

    print("\nTraining Collaborative Filtering (SVD) model...")
    cf_svd_model = CollaborativeFilteringRecommender(algorithm="svd")
    cf_svd_model.fit(books_df, ratings_df)
    print(f"  Done in {cf_svd_model.train_time:.3f}s")

    print("\nTraining Collaborative Filtering (KNN) model...")
    cf_knn_model = CollaborativeFilteringRecommender(algorithm="knn")
    cf_knn_model.fit(books_df, ratings_df)
    print(f"  Done in {cf_knn_model.train_time:.3f}s")

    # Run comparison if not already cached
    results_path = "data/comparison_results.json"
    if os.path.exists(results_path):
        with open(results_path) as f:
            comparison_results = json.load(f)
        print("\nLoaded cached comparison results.")
    else:
        print("\nRunning algorithm comparison (first time)...")
        comparison_results = run_full_comparison(books_df, ratings_df)

    print("\n=== All models ready! Server starting... ===\n")


# ---- Routes ----

@app.route("/")
def index():
    """Home page with system overview."""
    genres = sorted(books_df["genre"].unique().tolist())
    users = sorted(ratings_df["user_id"].unique().tolist())
    stats = {
        "total_books": len(books_df),
        "total_ratings": len(ratings_df),
        "total_users": ratings_df["user_id"].nunique(),
        "total_genres": len(genres),
        "avg_rating": round(ratings_df["rating"].mean(), 2),
    }
    return render_template("index.html", stats=stats, genres=genres, users=users)


@app.route("/recommend", methods=["POST"])
def recommend():
    """Get book recommendations for a user."""
    user_id = int(request.form.get("user_id", 1))
    algorithm = request.form.get("algorithm", "content_based")
    n = int(request.form.get("n_recommendations", 10))

    if algorithm == "content_based":
        recs = cb_model.recommend(user_id, n)
        algo_name = "Content-Based Filtering (TF-IDF + Cosine Similarity)"
    elif algorithm == "collaborative_svd":
        recs = cf_svd_model.recommend(user_id, n)
        algo_name = "Collaborative Filtering (SVD - Matrix Factorization)"
    elif algorithm == "collaborative_knn":
        recs = cf_knn_model.recommend(user_id, n)
        algo_name = "Collaborative Filtering (KNN)"
    else:
        recs = cb_model.recommend(user_id, n)
        algo_name = "Content-Based Filtering"

    # Get user's rated books for context
    user_ratings = ratings_df[ratings_df["user_id"] == user_id].merge(
        books_df[["book_id", "title", "author", "genre"]], on="book_id", how="left"
    ).sort_values("rating", ascending=False)

    return render_template(
        "recommendations.html",
        recommendations=recs.to_dict("records") if not recs.empty else [],
        user_id=user_id,
        algorithm=algorithm,
        algo_name=algo_name,
        user_ratings=user_ratings.to_dict("records"),
        n=n
    )


@app.route("/compare")
def compare():
    """Algorithm comparison dashboard."""
    return render_template("compare.html", results=comparison_results)


@app.route("/run_comparison")
def run_comparison_route():
    """Re-run the full comparison (can be triggered from UI)."""
    global comparison_results
    comparison_results = run_full_comparison(books_df, ratings_df)
    return render_template("compare.html", results=comparison_results)


@app.route("/books")
def browse_books():
    """Browse all books in the dataset."""
    genre_filter = request.args.get("genre", "all")
    if genre_filter and genre_filter != "all":
        filtered = books_df[books_df["genre"] == genre_filter]
    else:
        filtered = books_df
    genres = sorted(books_df["genre"].unique().tolist())
    return render_template(
        "books.html",
        books=filtered.to_dict("records"),
        genres=genres,
        selected_genre=genre_filter
    )


@app.route("/book/<int:book_id>")
def book_detail(book_id):
    """Book detail page with similar books from content-based model."""
    book = books_df[books_df["book_id"] == book_id]
    if book.empty:
        return "Book not found", 404
    book = book.iloc[0].to_dict()

    similar = cb_model.get_similar_books(book_id, n=5)
    avg_rating = ratings_df[ratings_df["book_id"] == book_id]["rating"].mean()
    num_ratings = len(ratings_df[ratings_df["book_id"] == book_id])

    return render_template(
        "book_detail.html",
        book=book,
        similar_books=similar.to_dict("records") if not similar.empty else [],
        avg_rating=round(avg_rating, 2) if not pd.isna(avg_rating) else "N/A",
        num_ratings=num_ratings
    )


@app.route("/api/recommend/<int:user_id>")
def api_recommend(user_id):
    """API endpoint for recommendations (JSON)."""
    algo = request.args.get("algorithm", "content_based")
    n = int(request.args.get("n", 10))

    if algo == "content_based":
        recs = cb_model.recommend(user_id, n)
    elif algo == "collaborative_svd":
        recs = cf_svd_model.recommend(user_id, n)
    else:
        recs = cf_knn_model.recommend(user_id, n)

    return jsonify(recs.to_dict("records") if not recs.empty else [])


import pandas as pd

if __name__ == "__main__":
    initialize()
    app.run(debug=False, port=5000)
