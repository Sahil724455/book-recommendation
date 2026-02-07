"""
Collaborative Filtering Algorithm
===================================
Uses K-Nearest Neighbors (KNN) and Matrix Factorization (SVD)
implemented with scikit-learn and scipy to recommend books based on user behavior.

How it works:
1. Build a user-item rating matrix.
2. KNN: Find users with similar rating patterns -> recommend their liked books.
3. SVD: Decompose the rating matrix into latent factors -> predict unseen ratings.

Advantages:
- High accuracy with sufficient data
- Highly personalized recommendations
- Captures complex user preferences
- Widely used in industry (Netflix, Amazon)
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold


class CollaborativeFilteringRecommender:
    """Collaborative Filtering using KNN + SVD (scikit-learn / scipy)."""

    def __init__(self, algorithm="svd"):
        """
        Initialize with chosen algorithm.
        algorithm: 'svd', 'knn', or 'ensemble'
        """
        self.algorithm = algorithm
        self.books_df = None
        self.ratings_df = None
        self.train_time = 0
        self.predict_times = []

        # SVD components
        self.user_factors = None
        self.item_factors = None
        self.sigma = None
        self.svd_predictions = None  # Full predicted ratings matrix
        self.global_mean = 0

        # KNN components
        self.knn_model = None
        self.user_item_matrix = None  # Dense matrix for KNN
        self.user_id_map = {}    # user_id -> matrix row index
        self.item_id_map = {}    # book_id -> matrix col index
        self.reverse_user_map = {}
        self.reverse_item_map = {}

    def _build_user_item_matrix(self, ratings_df):
        """Build user-item rating matrix from ratings dataframe."""
        # Create mappings
        unique_users = sorted(ratings_df["user_id"].unique())
        unique_items = sorted(ratings_df["book_id"].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.reverse_user_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.reverse_item_map = {idx: iid for iid, idx in self.item_id_map.items()}

        n_users = len(unique_users)
        n_items = len(unique_items)

        # Build dense matrix (fill missing with 0)
        matrix = np.zeros((n_users, n_items))
        for _, row in ratings_df.iterrows():
            u_idx = self.user_id_map[row["user_id"]]
            i_idx = self.item_id_map[row["book_id"]]
            matrix[u_idx, i_idx] = row["rating"]

        return matrix

    def fit(self, books_df, ratings_df):
        """
        Train the collaborative filtering model(s).
        """
        start_time = time.time()

        self.books_df = books_df.copy()
        self.ratings_df = ratings_df.copy()

        # Build user-item matrix
        self.user_item_matrix = self._build_user_item_matrix(ratings_df)
        self.global_mean = ratings_df["rating"].mean()

        n_users, n_items = self.user_item_matrix.shape

        # --- Train SVD (Matrix Factorization) ---
        if self.algorithm in ("svd", "ensemble"):
            # Demean the matrix (subtract user means for better SVD)
            user_ratings_mean = np.mean(self.user_item_matrix, axis=1).reshape(-1, 1)
            matrix_demeaned = self.user_item_matrix - user_ratings_mean

            # SVD decomposition
            n_factors = min(50, min(n_users, n_items) - 1)
            U, sigma, Vt = svds(csr_matrix(matrix_demeaned), k=n_factors)

            sigma = np.diag(sigma)
            self.svd_predictions = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean
            # Clip to valid rating range
            self.svd_predictions = np.clip(self.svd_predictions, 1.0, 5.0)

            self.user_factors = U
            self.item_factors = Vt.T
            self.sigma = sigma

        # --- Train KNN ---
        if self.algorithm in ("knn", "ensemble"):
            self.knn_model = NearestNeighbors(
                n_neighbors=min(21, n_users),  # k+1 because query includes itself
                metric="cosine",
                algorithm="brute"
            )
            self.knn_model.fit(self.user_item_matrix)

        self.train_time = time.time() - start_time
        return self

    def _predict_svd(self, user_id, book_id):
        """Predict rating using SVD."""
        if user_id not in self.user_id_map or book_id not in self.item_id_map:
            return self.global_mean

        u_idx = self.user_id_map[user_id]
        i_idx = self.item_id_map[book_id]
        return float(self.svd_predictions[u_idx, i_idx])

    def _predict_knn(self, user_id, book_id):
        """Predict rating using KNN (user-based)."""
        if user_id not in self.user_id_map or book_id not in self.item_id_map:
            return self.global_mean

        u_idx = self.user_id_map[user_id]
        i_idx = self.item_id_map[book_id]

        # Find k nearest neighbors
        user_vector = self.user_item_matrix[u_idx].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(user_vector)

        # Weighted average of neighbors' ratings for this item
        weighted_sum = 0.0
        weight_total = 0.0

        for dist, neighbor_idx in zip(distances[0][1:], indices[0][1:]):  # Skip self
            similarity = 1 - dist  # Convert distance to similarity
            if similarity <= 0:
                continue
            neighbor_rating = self.user_item_matrix[neighbor_idx, i_idx]
            if neighbor_rating > 0:  # Only consider if neighbor rated this item
                weighted_sum += similarity * neighbor_rating
                weight_total += similarity

        if weight_total > 0:
            return np.clip(weighted_sum / weight_total, 1.0, 5.0)
        return self.global_mean

    def predict_rating(self, user_id, book_id):
        """Predict a user's rating for a specific book."""
        if self.algorithm == "ensemble":
            svd_pred = self._predict_svd(user_id, book_id)
            knn_pred = self._predict_knn(user_id, book_id)
            return 0.6 * svd_pred + 0.4 * knn_pred
        elif self.algorithm == "svd":
            return self._predict_svd(user_id, book_id)
        else:
            return self._predict_knn(user_id, book_id)

    def recommend(self, user_id, n_recommendations=10):
        """
        Generate recommendations for a user.
        1. Find all books the user hasn't rated.
        2. Predict ratings for all unseen books.
        3. Return top-N books by predicted rating.
        """
        start_time = time.time()

        # Books the user has already rated
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        rated_books = set(user_ratings["book_id"].values)
        all_books = set(self.books_df["book_id"].values)
        unseen_books = all_books - rated_books

        if not unseen_books:
            self.predict_times.append(time.time() - start_time)
            return pd.DataFrame()

        # Predict ratings for all unseen books
        predictions = []
        for book_id in unseen_books:
            pred_rating = self.predict_rating(user_id, book_id)
            predictions.append({
                "book_id": book_id,
                "predicted_rating": round(pred_rating, 2)
            })

        results = pd.DataFrame(predictions)
        results = results.sort_values("predicted_rating", ascending=False).head(n_recommendations)

        # Merge with book details
        results = results.merge(
            self.books_df[["book_id", "title", "author", "genre", "year", "description"]],
            on="book_id", how="left"
        )
        results["score"] = results["predicted_rating"]

        self.predict_times.append(time.time() - start_time)
        return results

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

    def cross_validate_model(self, cv=5):
        """
        Perform k-fold cross-validation.
        Returns average RMSE and MAE across folds.
        """
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        rmse_scores = []
        mae_scores = []

        ratings_array = self.ratings_df.values

        for train_idx, test_idx in kf.split(ratings_array):
            train_data = pd.DataFrame(
                ratings_array[train_idx],
                columns=self.ratings_df.columns
            )
            test_data = pd.DataFrame(
                ratings_array[test_idx],
                columns=self.ratings_df.columns
            )

            # Ensure correct dtypes
            train_data["user_id"] = train_data["user_id"].astype(int)
            train_data["book_id"] = train_data["book_id"].astype(int)
            train_data["rating"] = train_data["rating"].astype(float)
            test_data["user_id"] = test_data["user_id"].astype(int)
            test_data["book_id"] = test_data["book_id"].astype(int)
            test_data["rating"] = test_data["rating"].astype(float)

            # Train a fresh model on this fold
            fold_model = CollaborativeFilteringRecommender(algorithm=self.algorithm)
            fold_model.fit(self.books_df, train_data)

            # Evaluate on test fold
            preds = []
            actuals = []
            for _, row in test_data.iterrows():
                pred = fold_model.predict_rating(int(row["user_id"]), int(row["book_id"]))
                preds.append(pred)
                actuals.append(row["rating"])

            preds = np.array(preds)
            actuals = np.array(actuals)

            rmse_scores.append(np.sqrt(np.mean((preds - actuals) ** 2)))
            mae_scores.append(np.mean(np.abs(preds - actuals)))

        return {
            "cv_rmse": round(np.mean(rmse_scores), 4),
            "cv_mae": round(np.mean(mae_scores), 4),
            "cv_rmse_std": round(np.std(rmse_scores), 4),
            "cv_mae_std": round(np.std(mae_scores), 4),
        }

    def get_similar_users(self, user_id, n=5):
        """Find n most similar users using KNN model."""
        if self.knn_model is None or user_id not in self.user_id_map:
            return []

        u_idx = self.user_id_map[user_id]
        user_vector = self.user_item_matrix[u_idx].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(user_vector, n_neighbors=n + 1)

        similar_users = []
        for idx in indices[0][1:]:  # Skip self
            similar_users.append(self.reverse_user_map[idx])
        return similar_users
