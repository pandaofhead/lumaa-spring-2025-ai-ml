import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:
    def __init__(self, data_path):
        """Initialize the recommender system"""
        self.df = pd.read_csv(data_path)
        self.vectorizer = None
        self.tfidf_matrix = None
    def preprocess_data(self):
        """Clean and prepare review data"""
        # Enhanced preprocessing
        self.df['review'] = self.df['review'].fillna('')
        self.df['review'] = self.df['review'].str.replace('<br />', ' ')
        self.df['review'] = self.df['review'].str.lower()
        
        # Add more cleaning steps
        self.df['review'] = self.df['review'].str.replace('[^\w\s]', '')  # Remove punctuation
        self.df['review'] = self.df['review'].str.strip()  # Remove extra whitespace
        
    def create_vectors(self):
        """Create TF-IDF vectors from review data"""
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,           # Ignore terms that appear in less than 2 documents
            max_df=0.95,        # Ignore terms that appear in more than 95% of documents
            analyzer='word'
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['review'])
    
    def get_recommendations(self, user_input, n_recommendations=5, min_similarity=0.0):
        """Get movie recommendations based on user input"""
        recommendations = []
        user_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        
        # Get top N similar movies above minimum similarity threshold
        top_indices = similarities[0].argsort()[-n_recommendations*2:][::-1]  # Get more candidates
        
        for idx in top_indices:
            similarity = similarities[0][idx]
            if similarity < min_similarity:
                continue
                
            if len(recommendations) >= n_recommendations:
                break
                
            recommendations.append({
                'review': self.df.iloc[idx]['review'][:200] + "...",
                'sentiment': self.df.iloc[idx]['sentiment'],
                'similarity_score': similarity
            })
            
        return recommendations

def main():
    # Initialize recommender
    recommender = MovieRecommender('data/IMDB Dataset.csv')
    
    # Prepare the system
    recommender.preprocess_data()
    recommender.create_vectors()
    
    # Example recommendation
    user_input = "A fantastic sci-fi movie with great special effects and compelling story"
    recommendations = recommender.get_recommendations(user_input)
    
    # Print recommendations
    print("\nYour input:", user_input)
    print("\nTop Similar Reviews:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Sentiment: {rec['sentiment']}")
        print(f"Similarity Score: {rec['similarity_score']:.2f}")
        print(f"Review: {rec['review']}")

if __name__ == "__main__":
    main()