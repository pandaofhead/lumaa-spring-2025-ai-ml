# Content-Based Movie Recommendation System

A simple content-based recommendation system that suggests similar movie reviews based on user text input. The system uses TF-IDF vectorization and cosine similarity to match user preferences with movie reviews from the IMDB dataset.

## 1. Dataset

- Using the IMDB Dataset containing 50K movie reviews
- Dataset contains two columns:
  - `review`: Text review of the movie
  - `sentiment`: Sentiment analysis of the review (positive/negative)
- Dataset size: 50,000 reviews
- Source: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

### Dataset Setup

1. Download the IMDB Dataset from Kaggle
2. Place the 'IMDB Dataset.csv' file in the `data/` directory of this project

## 2. Setup Instructions

1. **Environment Setup**

   ```bash
   # Create conda environment
   conda create -n movie-rec python=3.8
   conda activate movie-rec

   # Install required packages
   pip install pandas numpy scikit-learn
   ```

2. **Project Structure**
   ```
   project/
   ├── data/
   │   └── IMDB Dataset.csv
   ├── src/
   │   ├── recommender.py
   │   └── cli.py
   ├── notebooks/
   │   └── explore.ipynb
   └── README.md
   ```

## 3. Running the Code

### Basic Usage

```bash
python src/recommender.py
```

### Cli Usage

```bash
python src/cli.py --input "A fantastic sci-fi movie with great special effects and compelling story" --num 5 --threshold 0.1
```

### Example Results

```
Input: "I love sci-fi movies with great special effects"

Sample Output:
Top Similar Reviews:
Sentiment: positive
Similarity Score: 0.85
Review: "Amazing sci-fi effects and stunning visuals..."
Sentiment: positive
Similarity Score: 0.76
Review: "The special effects were groundbreaking..."
```

## 4. Demo Video

[Demo Video Link](https://www.loom.com/share/4c5dfa8411e145acae507f33800ac7bd?sid=827e4aaf-d71e-4500-9194-a5e36d12bb1e)

## 5. Implementation Details

### Technical Approach

1. **Text Preprocessing**

   - Convert to lowercase
   - Remove HTML tags
   - Remove punctuation
   - Strip extra whitespace

2. **Content-Based Filtering**

   - TF-IDF Vectorization
   - Cosine Similarity matching
   - Configurable similarity threshold

## Code Organization

- `src/recommender.py`: Core recommendation system
- `src/cli.py`: Command-line interface
- `notebooks/explore.ipynb`: Data exploration and testing

## Dependencies

- Python 3.8+
- pandas
- numpy
- scikit-learn

## Salary Expectation

4000 USD per month

## Future Improvements

- Add genre-based filtering
- Implement more advanced text preprocessing

## Author

Hongjin Quan

## License

MIT License
