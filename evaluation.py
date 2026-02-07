"""
Evaluation & Comparison Module
================================
Compares Content-Based Filtering and Collaborative Filtering algorithms
across multiple metrics: RMSE, MAE, training time, prediction time,
and cross-validation performance.

Generates comparison charts saved as static images for the web dashboard.
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from content_based import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender


# Set style for charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

CHARTS_DIR = "static/charts"


def ensure_charts_dir():
    os.makedirs(CHARTS_DIR, exist_ok=True)


def run_full_comparison(books_df, ratings_df, test_size=0.2, seed=42):
    """
    Run a complete comparison between Content-Based and Collaborative Filtering.
    Returns a dictionary with all metrics and chart paths.
    """
    ensure_charts_dir()

    # --- Split data ---
    train_ratings, test_ratings = train_test_split(
        ratings_df, test_size=test_size, random_state=seed
    )
    print(f"Train size: {len(train_ratings)}, Test size: {len(test_ratings)}")

    results = {}

    # =============================================
    # Algorithm 1: Content-Based Filtering
    # =============================================
    print("\n--- Training Content-Based Filtering ---")
    cb_model = ContentBasedRecommender()
    cb_model.fit(books_df, train_ratings)
    print(f"  Training time: {cb_model.train_time:.4f}s")

    print("  Evaluating on test set...")
    cb_metrics = cb_model.evaluate(test_ratings)
    results["content_based"] = {
        "name": "Content-Based Filtering",
        "method": "TF-IDF + Cosine Similarity",
        **cb_metrics
    }
    print(f"  RMSE: {cb_metrics['rmse']}, MAE: {cb_metrics['mae']}")

    # =============================================
    # Algorithm 2: Collaborative Filtering (SVD)
    # =============================================
    print("\n--- Training Collaborative Filtering (SVD) ---")
    cf_svd_model = CollaborativeFilteringRecommender(algorithm="svd")
    cf_svd_model.fit(books_df, train_ratings)
    print(f"  Training time: {cf_svd_model.train_time:.4f}s")

    print("  Evaluating on test set...")
    cf_svd_metrics = cf_svd_model.evaluate(test_ratings)

    print("  Running 5-fold cross-validation...")
    cf_svd_cv = cf_svd_model.cross_validate_model(cv=5)

    results["collaborative_svd"] = {
        "name": "Collaborative Filtering (SVD)",
        "method": "Matrix Factorization (SVD)",
        **cf_svd_metrics,
        **cf_svd_cv
    }
    print(f"  RMSE: {cf_svd_metrics['rmse']}, MAE: {cf_svd_metrics['mae']}")
    print(f"  CV RMSE: {cf_svd_cv.get('cv_rmse', 'N/A')}")

    # =============================================
    # Algorithm 2b: Collaborative Filtering (KNN)
    # =============================================
    print("\n--- Training Collaborative Filtering (KNN) ---")
    cf_knn_model = CollaborativeFilteringRecommender(algorithm="knn")
    cf_knn_model.fit(books_df, train_ratings)
    print(f"  Training time: {cf_knn_model.train_time:.4f}s")

    print("  Evaluating on test set...")
    cf_knn_metrics = cf_knn_model.evaluate(test_ratings)

    print("  Running 5-fold cross-validation...")
    cf_knn_cv = cf_knn_model.cross_validate_model(cv=5)

    results["collaborative_knn"] = {
        "name": "Collaborative Filtering (KNN)",
        "method": "K-Nearest Neighbors",
        **cf_knn_metrics,
        **cf_knn_cv
    }
    print(f"  RMSE: {cf_knn_metrics['rmse']}, MAE: {cf_knn_metrics['mae']}")
    print(f"  CV RMSE: {cf_knn_cv.get('cv_rmse', 'N/A')}")

    # =============================================
    # Generate comparison charts
    # =============================================
    print("\n--- Generating comparison charts ---")
    chart_paths = generate_charts(results)
    results["charts"] = chart_paths

    # Determine winner
    results["winner"] = determine_winner(results)

    # Save results to JSON
    with open("data/comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nComparison complete. Winner: {results['winner']['name']}")
    return results


def determine_winner(results):
    """Determine which algorithm is best overall."""
    algos = ["content_based", "collaborative_svd", "collaborative_knn"]
    scores = {}

    for algo in algos:
        r = results[algo]
        # Lower RMSE is better, lower train time is better, lower prediction time is better
        score = 0
        for other in algos:
            if other == algo:
                continue
            o = results[other]
            if r["rmse"] < o["rmse"]:
                score += 3  # Accuracy is most important
            if r["mae"] < o["mae"]:
                score += 2
            if r["train_time"] < o["train_time"]:
                score += 1
            if r["avg_prediction_time"] < o["avg_prediction_time"]:
                score += 1
        scores[algo] = score

    best = max(scores, key=scores.get)
    return {
        "key": best,
        "name": results[best]["name"],
        "reason": f"Best overall score ({scores[best]} points) considering accuracy (RMSE/MAE) and efficiency (train/predict time).",
        "scores": scores
    }


def generate_charts(results):
    """Generate all comparison charts."""
    chart_paths = {}

    algos = ["content_based", "collaborative_svd", "collaborative_knn"]
    names = [results[a]["name"] for a in algos]
    short_names = ["Content-Based\n(TF-IDF)", "Collaborative\n(SVD)", "Collaborative\n(KNN)"]
    colors = ["#4F46E5", "#059669", "#D97706"]

    # 1. RMSE Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    rmse_vals = [results[a]["rmse"] for a in algos]
    bars = ax.bar(short_names, rmse_vals, color=colors, edgecolor="white", linewidth=2)
    ax.set_title("RMSE Comparison (Lower is Better)", fontweight="bold", pad=15)
    ax.set_ylabel("RMSE")
    for bar, val in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(rmse_vals) * 1.2)
    plt.tight_layout()
    path = f"{CHARTS_DIR}/rmse_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_paths["rmse"] = path

    # 2. MAE Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    mae_vals = [results[a]["mae"] for a in algos]
    bars = ax.bar(short_names, mae_vals, color=colors, edgecolor="white", linewidth=2)
    ax.set_title("MAE Comparison (Lower is Better)", fontweight="bold", pad=15)
    ax.set_ylabel("MAE")
    for bar, val in zip(bars, mae_vals):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(mae_vals) * 1.2)
    plt.tight_layout()
    path = f"{CHARTS_DIR}/mae_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_paths["mae"] = path

    # 3. Training Time Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    train_times = [results[a]["train_time"] for a in algos]
    bars = ax.bar(short_names, train_times, color=colors, edgecolor="white", linewidth=2)
    ax.set_title("Training Time Comparison (Lower is Better)", fontweight="bold", pad=15)
    ax.set_ylabel("Time (seconds)")
    for bar, val in zip(bars, train_times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f"{val:.4f}s", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(train_times) * 1.3)
    plt.tight_layout()
    path = f"{CHARTS_DIR}/training_time_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_paths["training_time"] = path

    # 4. Average Prediction Time Comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    pred_times = [results[a]["avg_prediction_time"] * 1000 for a in algos]  # Convert to ms
    bars = ax.bar(short_names, pred_times, color=colors, edgecolor="white", linewidth=2)
    ax.set_title("Avg Prediction Time (Lower is Better)", fontweight="bold", pad=15)
    ax.set_ylabel("Time (milliseconds)")
    for bar, val in zip(bars, pred_times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                f"{val:.3f}ms", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(pred_times) * 1.3)
    plt.tight_layout()
    path = f"{CHARTS_DIR}/prediction_time_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_paths["prediction_time"] = path

    # 5. Combined Radar/Overview Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 5a: Accuracy metrics grouped bar chart
    x = np.arange(len(short_names))
    width = 0.35
    axes[0].bar(x - width / 2, rmse_vals, width, label="RMSE", color="#4F46E5", alpha=0.85)
    axes[0].bar(x + width / 2, mae_vals, width, label="MAE", color="#059669", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, fontsize=10)
    axes[0].set_title("Accuracy Metrics", fontweight="bold")
    axes[0].set_ylabel("Error Value")
    axes[0].legend()

    # 5b: Efficiency metrics grouped bar chart
    axes[1].bar(x - width / 2, train_times, width, label="Train Time (s)", color="#D97706", alpha=0.85)
    axes[1].bar(x + width / 2, pred_times, width, label="Predict Time (ms)", color="#DC2626", alpha=0.85)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=10)
    axes[1].set_title("Efficiency Metrics", fontweight="bold")
    axes[1].set_ylabel("Time")
    axes[1].legend()

    plt.suptitle("Algorithm Comparison Overview", fontweight="bold", fontsize=15, y=1.02)
    plt.tight_layout()
    path = f"{CHARTS_DIR}/overview_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_paths["overview"] = path

    # 6. Heatmap of metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics_data = {
        "RMSE": rmse_vals,
        "MAE": mae_vals,
        "Train Time (s)": train_times,
        "Predict Time (ms)": pred_times,
    }
    df_heat = pd.DataFrame(metrics_data, index=[n.replace("\n", " ") for n in short_names])
    # Normalize for heatmap (0-1 scale per metric)
    df_norm = df_heat.copy()
    for col in df_norm.columns:
        col_min, col_max = df_norm[col].min(), df_norm[col].max()
        if col_max - col_min > 0:
            df_norm[col] = (df_norm[col] - col_min) / (col_max - col_min)
        else:
            df_norm[col] = 0.5

    sns.heatmap(df_norm, annot=df_heat.round(4), fmt="", cmap="RdYlGn_r",
                linewidths=2, ax=ax, cbar_kws={"label": "Normalized (lower=better)"})
    ax.set_title("Performance Heatmap (Values shown, colors normalized)", fontweight="bold", pad=15)
    plt.tight_layout()
    path = f"{CHARTS_DIR}/heatmap_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    chart_paths["heatmap"] = path

    return chart_paths


if __name__ == "__main__":
    from dataset import load_or_generate_data
    books_df, ratings_df = load_or_generate_data()
    results = run_full_comparison(books_df, ratings_df)
    print("\n===== FINAL RESULTS =====")
    for key in ["content_based", "collaborative_svd", "collaborative_knn"]:
        r = results[key]
        print(f"\n{r['name']}:")
        print(f"  RMSE: {r['rmse']}")
        print(f"  MAE:  {r['mae']}")
        print(f"  Train Time: {r['train_time']}s")
        print(f"  Avg Prediction Time: {r['avg_prediction_time']}s")
    print(f"\nWinner: {results['winner']['name']}")
    print(f"Reason: {results['winner']['reason']}")
