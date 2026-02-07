# BookMind AI - ML Book Recommendation System

A full-stack web application that provides personalized book recommendations using machine learning algorithms. Built with **Flask** (Python) on the backend and a modern dark-themed UI on the frontend.

## Features

- **Content-Based Filtering** — Uses TF-IDF vectorization and Cosine Similarity on book metadata (genre, author, description) to find similar books.
- **Collaborative Filtering (SVD)** — Matrix factorization via Singular Value Decomposition to uncover latent user/item factors and predict ratings.
- **Collaborative Filtering (KNN)** — K-Nearest Neighbors to find users with similar tastes and recommend their highly-rated books.
- **Algorithm Comparison Dashboard** — Side-by-side evaluation with RMSE, MAE, training time, prediction time, cross-validation, and auto-generated charts.
- **Browse & Search** — Filter 100 books across 10 genres, view detailed pages with similar-book suggestions.
- **REST API** — JSON endpoint for programmatic access to recommendations.

## Tech Stack

| Layer      | Technology                                            |
|------------|-------------------------------------------------------|
| Backend    | Python 3.13, Flask                                    |
| ML         | scikit-learn (TF-IDF, KNN, NearestNeighbors), SciPy (SVD), NumPy, Pandas |
| Charting   | Matplotlib, Seaborn                                   |
| Frontend   | Jinja2 Templates, HTML5, CSS3 (custom dark theme), Font Awesome |
| Data       | Synthetic dataset — 100 books, 10 users, ~300 ratings |

## Project Structure

```
book-recommendation/
├── app.py                     # Flask application & routes
├── dataset.py                 # Synthetic data generation (100 books, 10 genres)
├── content_based.py           # Content-Based Filtering (TF-IDF + Cosine Similarity)
├── collaborative_filtering.py # Collaborative Filtering (KNN + SVD)
├── evaluation.py              # Algorithm comparison & chart generation
├── requirements.txt           # Python dependencies
├── data/                      # Auto-generated CSV data & cached results
│   ├── books.csv
│   └── ratings.csv
├── static/
│   └── charts/                # Auto-generated comparison charts (PNG)
└── templates/                 # Jinja2 HTML templates
    ├── base.html              # Base layout with nav, styling, footer
    ├── index.html             # Home page with stats & recommendation form
    ├── books.html             # Browse books with genre filter
    ├── book_detail.html       # Single book detail + similar books
    ├── recommendations.html   # Personalized recommendation results
    └── compare.html           # Algorithm comparison dashboard
```

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/Sahil724455/book-recommendation.git
cd book-recommendation

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
python app.py
```

The server starts on **http://localhost:5000**. On first launch it will:
1. Generate the synthetic dataset (100 books, 10 users)
2. Train all three ML models
3. Run the algorithm comparison and generate charts

### API Usage

```bash
# Get 10 content-based recommendations for user 1
curl http://localhost:5000/api/recommend/1?algorithm=content_based&n=10

# Get SVD-based recommendations
curl http://localhost:5000/api/recommend/1?algorithm=collaborative_svd

# Get KNN-based recommendations
curl http://localhost:5000/api/recommend/1?algorithm=collaborative_knn
```

## Algorithms

### 1. Content-Based Filtering
- Combines book features (genre, author, description) into a single text field
- Applies **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization
- Computes **Cosine Similarity** between all book pairs
- Recommends books most similar to a user's highly-rated books

### 2. Collaborative Filtering — SVD
- Builds a **user-item rating matrix**
- Applies **Singular Value Decomposition** (scipy.sparse.linalg.svds) to extract latent factors
- Predicts ratings for unseen books from the reconstructed matrix
- Includes **5-fold cross-validation** for robust evaluation

### 3. Collaborative Filtering — KNN
- Uses **scikit-learn NearestNeighbors** with cosine distance
- Finds the k most similar users based on rating patterns
- Predicts ratings as a **weighted average** of neighbors' ratings
- Includes **5-fold cross-validation**

## Screenshots

The application features a modern dark-themed UI with:
- Home page with dataset stats and recommendation form
- Book browsing with genre filters
- Detailed book pages with content-based similar books
- Full algorithm comparison dashboard with charts

## License

This project is open source and available under the [MIT License](LICENSE).
