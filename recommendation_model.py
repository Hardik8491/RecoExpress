import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

class RecommendationModel:
    def __init__(self):
        with open('recommendation_model.pkl', 'rb') as f:
            self.vectorizer, self.nn_model, self.merged_df = pickle.load(f)

    def recommend_products(self, product_name, num_recommendations=5):
        query_vector = self.vectorizer.transform([product_name])
        distances, indices = self.nn_model.kneighbors(query_vector, n_neighbors=num_recommendations)
        recommendations = self.merged_df.iloc[indices[0]].copy()
        recommendations['distance'] = distances[0]
        return recommendations[['title', 'stars', 'price', 'category_name', 'distance']]
