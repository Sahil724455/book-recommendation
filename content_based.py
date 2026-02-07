"""
Content-Based Filtering Algorithm
==================================
Uses TF-IDF vectorization and Cosine Similarity to recommend books
based on the features (genre, author, description) of books a user liked.

How it works:
1. Combine book features (genre + author + description) into a single text field.
2. Convert text to numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
3. Compute cosine similarity between all book pairs.
4. For a given user, find their top-rated books.
5. Recommend books most similar to those top-rated books.

Advantages:
- Fast computation
- Works well with small datasets
- No cold-start problem for items
- Does not require many users
"""

import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold


class ContentBasedRecommender:
    """Content-Based Filtering using TF-IDF + Cosine Similarity."""

    def __init__(self):
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.books_df = None
        self.ratings_df = None
        self.tfidf_vectorizer = None
        self.train_time = 0
        self.predict_times = []

    def _create_feature_text(self, books_df):
        """Combine book features into a single text string for TF-IDF."""
        books_df = books_df.copy()
        books_df["features"] = (
            books_df["genre"].fillna("") + " " +
            books_df["genre"].fillna("") + " " +  # double-weight genre
            books_df["author"].fillna("") + " " +
            books_df["description"].fillna("")
        )
        return books_df

    def fit(self, books_df, ratings_df):
        """
        Train the content-based model.
        - Vectorize book features using TF-IDF
        - Compute cosine similarity matrix
        """
        start_time = time.time()

        self.books_df = self._create_feature_text(books_df.copy())
        self.ratings_df = ratings_df.copy()

        # TF-IDF Vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.books_df["features"]
        )

        # Cosine Similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        self.train_time = time.time() - start_time
        return self

    def recommend(self, user_id, n_recommendations=10):
        """
        Generate recommendations for a user.
        1. Find books the user rated highly (>= 3.5).
        2. For each liked book, find the most similar books.
        3. Aggregate similarity scores and return top-N unseen books.
        """
        start_time = time.time()

        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            self.predict_times.append(time.time() - start_time)
            return pd.DataFrame()

        # Books the user liked (rating >= 3.5)
        liked_books = user_ratings[user_ratings["rating"] >= 3.5]["book_id"].values
        rated_books = set(user_ratings["book_id"].values)

        if len(liked_books) == 0:
            # If no highly-rated books, use all rated books
            liked_books = user_ratings["book_id"].values

        # Compute aggregated similarity scores
        book_ids = self.books_df["book_id"].values
        similarity_scores = np.zeros(len(self.books_df))

        for book_id in liked_books:
            idx = self.books_df[self.books_df["book_id"] == book_id].index
            if len(idx) > 0:
                idx = idx[0]
                user_rating = user_ratings[user_ratings["book_id"] == book_id]["rating"].values[0]
                # Weight similarity by user's rating
                similarity_scores += self.cosine_sim[idx] * (user_rating / 5.0)

        # Build results dataframe
        results = pd.DataFrame({
            "book_id": book_ids,
            "score": similarity_scores
        })

        # Exclude already-rated books
        results = results[~results["book_id"].isin(rated_books)]
        results = results.sort_values("score", ascending=False).head(n_recommendations)

        # Merge with book details
        results = results.merge(self.books_df[["book_id", "title", "author", "genre", "year", "description"]],
                                on="book_id", how="left")

        # Normalize score to 0-5 range for display
        if results["score"].max() > 0:
            results["predicted_rating"] = (results["score"] / results["score"].max()) * 5.0
        else:
            results["predicted_rating"] = 0.0

        self.predict_times.append(time.time() - start_time)
        return results

    def predict_rating(self, user_id, book_id):
        """Predict a user's rating for a specific book."""
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return 3.0  # Default

        liked_books = user_ratings[user_ratings["rating"] >= 3.0]["book_id"].values
        if len(liked_books) == 0:
            liked_books = user_ratings["book_id"].values

        target_idx = self.books_df[self.books_df["book_id"] == book_id].index
        if len(target_idx) == 0:
            return 3.0
        target_idx = target_idx[0]

        weighted_sum = 0
        weight_total = 0

        for bid in liked_books:
            idx = self.books_df[self.books_df["book_id"] == bid].index
            if len(idx) > 0:
                idx = idx[0]
                sim = self.cosine_sim[target_idx][idx]
                actual_rating = user_ratings[user_ratings["book_id"] == bid]["rating"].values[0]
                weighted_sum += sim * actual_rating
                weight_total += abs(sim)

        if weight_total == 0:
            return 3.0
        return np.clip(weighted_sum / weight_total, 1.0, 5.0)

    def evaluate(self, test_ratings):
        """
        Evaluate model on test data.
        Returns RMSE, MAE, and average prediction time.
        """
        predictions = []
        actuals = []
        pred_times = []

        for _, row in test_ratings.iterrows():
            start = time.time()
            pred = self.predict_rating(int(row["user_id"]), int(row["book_id"]))
            pred_times.append(time.time() - start)
            predictions.append(pred)
            actuals.append(row["rating"])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        avg_pred_time = np.mean(pred_times) if pred_times else 0

        return {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "train_time": round(self.train_time, 4),
            "avg_prediction_time": round(avg_pred_time, 6),
            "total_predictions": len(predictions)
        }

    def get_similar_books(self, book_id, n=5):
        """Find n most similar books to a given book."""
        idx = self.books_df[self.books_df["book_id"] == book_id].index
        if len(idx) == 0:
            return pd.DataFrame()
        idx = idx[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n + 1]  # Exclude itself

        book_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        result = self.books_df.iloc[book_indices][["book_id", "title", "author", "genre", "year"]].copy()
        result["similarity_score"] = scores
        return result
