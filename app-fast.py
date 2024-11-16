from fastapi import FastAPI, HTTPException, Query
from typing import List
import joblib
import pandas as pd
from pydantic import BaseModel
import dill
from fastapi.middleware.cors import CORSMiddleware
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize FastAPI
app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the pickled function



# Load your data
products = dill.load(open("products_list.pkl", "rb"))
sentiments = dill.load(open("sentiments.pkl", "rb"))

# top_rated_products = dill.load(open('top_rated_products.pkl'))

# Define the root endpoint
@app.get("/")
async def read_root():
    top_rated_products = process_dataframe(products)
    return {"products": top_rated_products}

@app.get("/products/{product_id}")
def get_product(product_id: int):
    product = products[products["id"] == product_id].to_dict(orient="records")
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product[0]

@app.get("/recommendations/{product_id}")
def get_recommendations(product_id: int, num_recommendations: int = 5):
    if product_id not in products['id'].values:
        raise HTTPException(status_code=404, detail="Product not found")
    
    recommendations = recommend_items(product_id, num_recommendations)
    return {"recommendations": recommendations}

@app.get("/reviews/{product_id}")
def get_reviews_by_sentiment(product_id: int):
    # Filter reviews for the specific product
    product_reviews = sentiments[sentiments["id"] == product_id][["review", "sentiment"]]
    
    if product_reviews.empty:
        raise HTTPException(status_code=404, detail="No reviews found for this product")
    
    # Group reviews by sentiment
    grouped_reviews = product_reviews.groupby("sentiment")["review"].apply(list).to_dict()
    
    return grouped_reviews

# Assuming df is your DataFrame
def process_dataframe(df):
    # Group by category and sort by ratings to get top products in each category
    top_rated_by_category = (
        df.sort_values(by="ratings_count", ascending=False)
          .groupby("Category")
          .head(5)  # Adjust this number based on the number of top products you want per category
    )
    return top_rated_by_category.to_dict(orient="records")

# Build a recommendation Function
def recommend_items(product_id, num_recommendations=5):
    # Combine the features
    products['combined_features'] = (products['product_name'].fillna('') + products['brand'].fillna('') + products['review_title'].fillna('') + products['review'].fillna(''))
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(products['combined_features'])

    # Set the number of latent features (e.g., 50)
    n_components = 50

    # Initialize and apply TruncatedSVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    item_features_matrix = svd.fit_transform(tfidf_matrix)
    
    # Calculate cosine similarity on the SVD-transformed matrix
    cosine_sim = cosine_similarity(item_features_matrix)
    # Convert to DataFrame for ease of use
    cosine_sim_df = pd.DataFrame(cosine_sim, index=products['id'], columns=products['id'])
    # Check if the product exists in the similarity DataFrame
    if product_id not in cosine_sim_df.columns:
        print(f"Product '{product_id}' not found in dataset.")
        return []
    
    # Get similarity scores for the product and sort by descending order
    sim_scores = cosine_sim_df[product_id].sort_values(ascending=False)
    
    # Exclude the product itself from recommendations
    sim_scores = sim_scores.drop(product_id)
    
    # Get the top recommendations based on similarity scores
    top_recommendations_ids = sim_scores.head(num_recommendations).index.tolist()

    # Retrieve the recommended product details
    recommended_products = products[products['id'].isin(top_recommendations_ids)].to_dict(orient='records')

    return recommended_products

