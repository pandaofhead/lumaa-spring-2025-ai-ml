import argparse
from recommender import MovieRecommender

def main():
    parser = argparse.ArgumentParser(description='Movie Recommendation System')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input text description for recommendation')
    parser.add_argument('--num', '-n', type=int, default=5,
                       help='Number of recommendations to return')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,
                       help='Minimum similarity threshold')
    
    args = parser.parse_args()
    
    # Initialize recommender
    recommender = MovieRecommender('data/IMDB Dataset.csv')
    recommender.preprocess_data()
    recommender.create_vectors()
    
    # Get recommendations
    recommendations = recommender.get_recommendations(
        args.input, 
        n_recommendations=args.num,
        min_similarity=args.threshold
    )
    
    # Print recommendations
    print(f"\nYour input: {args.input}")
    print("\nTop Similar Reviews:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. Sentiment: {rec['sentiment']}")
        print(f"Similarity Score: {rec['similarity_score']:.2f}")
        print(f"Review: {rec['review']}")

if __name__ == "__main__":
    main()