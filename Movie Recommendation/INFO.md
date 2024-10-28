# 1. Data Preprocessing
## It is simple but I have to go through a simple level of `Data-processing`
- In this project I have really used my little brain, by zipping the resource file and then upload it to the [colab](https://colab.research.google.com/) and then unzipping em' using `!unzip filename.zip`.
- Then I have accessed it using `pd.read_csv('filename.csv')`. I have two files `tmdb_5000_credits.csv` and `tmdb_5000_movies.csv`.
- For `credits.info()`: 
	```
	<class 'pandas.core.frame.DataFrame'>
	RangeIndex: 4803 entries, 0 to 4802
	Data columns (total 4 columns):
	 #   Column    Non-Null Count  Dtype 
	---  ------    --------------  ----- 
	 0   movie_id  4803 non-null   int64 
	 1   title     4803 non-null   object
	 2   cast      4803 non-null   object
	 3   crew      4803 non-null   object
	dtypes: int64(1), object(3)
	memory usage: 150.2+ KB
	```
- And for `movies.info()`:
	```
	<class 'pandas.core.frame.DataFrame'>
	RangeIndex: 4803 entries, 0 to 4802
	Data columns (total 20 columns):
	 #   Column                Non-Null Count  Dtype  
	---  ------                --------------  -----  
	 0   budget                4803 non-null   int64  
	 1   genres                4803 non-null   object 
	 2   homepage              1712 non-null   object 
	 3   id                    4803 non-null   int64  
	 4   keywords              4803 non-null   object 
	 5   original_language     4803 non-null   object 
	 6   original_title        4803 non-null   object 
	 7   overview              4800 non-null   object 
	 8   popularity            4803 non-null   float64
	 9   production_companies  4803 non-null   object 
	 10  production_countries  4803 non-null   object 
	 11  release_date          4802 non-null   object 
	 12  revenue               4803 non-null   int64  
	 13  runtime               4801 non-null   float64
	 14  spoken_languages      4803 non-null   object 
	 15  status                4803 non-null   object 
	 16  tagline               3959 non-null   object 
	 17  title                 4803 non-null   object 
	 18  vote_average          4803 non-null   float64
	 19  vote_count            4803 non-null   int64  
	dtypes: float64(3), int64(4), object(13)
	memory usage: 750.6+ KB
	```
- Now as I have some many non-relatable columns that ain't gonna help my model recommend any movie; for example let's take budget in consideration; " **Godzilla: Minus one** " is a phenomenal movie under of budget of  **$15 Million**.
- So I've then merged both the `DataFrames` and the final one contains columns like `['movie_id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']`.
- After that I have converted each and every items of `['genres', 'keywords', 'overview', 'cast', 'crew']` columns from `json` to `list`.
- Then I've combined the above mentioned columns to `tags` using:
	```python
	movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
	```
# 2. Vectorization of each movie
-  I am using the `CountVectorizer` from `scikit-learn` to create a matrix of token counts (word frequencies) from the `tags` column in the `movies` `DataFrame`.
	```python
	from sklearn.feature_extraction.text import CountVectorizer
	
	cv = CountVectorizer(max_features=5000, stop_words='english')
	vectors = cv.fit_transform(movies['tags']).toarray()
	vectors, vectors[0]
	```
	```
	(array([[0, 0, 0, ..., 0, 0, 0],
	        [0, 0, 0, ..., 0, 0, 0],
	        [0, 0, 0, ..., 0, 0, 0],
	        ...,
	        [0, 0, 0, ..., 0, 0, 0],
	        [0, 0, 0, ..., 0, 0, 0],
	        [0, 0, 0, ..., 0, 0, 0]]),
	 array([0, 0, 0, ..., 0, 0, 0]))
	```
- **Explanation: ** 
	1. **`CountVectorizer(max_features=5000, stop_words='english')`**:
	    - This creates a `CountVectorizer` instance that will tokenize the text data in `movies['tags']`.
	    - `max_features=5000` limits the vocabulary to the 5,000 most frequently occurring words.
	    - `stop_words='english'` removes common English stop words like "the," "and," etc., from the text to focus on more meaningful words.
	2. **`vectors = cv.fit_transform(movies['tags']).toarray()`**:
	    - The `fit_transform` method tokenizes and builds a vocabulary from `movies['tags']`, then transforms each document (entry) in `tags` into a vector representing word counts.
	    - `toarray()` converts the resulting sparse matrix to a dense NumPy array.
	    - Each row of `vectors` represents a document in `movies['tags']`, and each column corresponds to a word in the vocabulary (up to 5,000 columns).
# 3. Stemming using `PorterStemmer`:
- I am applying **stemming** to the words in the `tags` of the movie description. <u>Stemming reduces each word to its base or root form by removing suffixes.</u>
- `PorterStemmer` from `nltk` is used to stem each word in `movies['tags']`.
# 4. Cosine Similarity:
- After vectorizing text data (e.g., using TF-IDF or Count Vectorizer), cosine similarity is a common metric used to measure the similarity between vectors(that's what I am working for... the similarities between movies). Cosine similarity calculates the cosine of the angle between two vectors, returning a value between 0 and 1, where:
	- 1 indicates identical vectors (highest similarity).
	- 0 indicates no similarity.
- Cosine similarity is particularly useful for text data, as it captures similarity based on word frequency and ignores differences in text length, making it ideal for tasks like document comparison, clustering, or recommendation systems.
	```python
	from sklearn.metrics.pairwise import cosine_similarity
	
	similarity = cosine_similarity(vectors)
	
	def recommend(movie_name):
		movie_idx = movies[movies['title'] == movie_name].index[0]
		distances = similarity[movie_idx]
		movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
	
		for i in movies_list:
		print(movies.iloc[i[0]].title)
	
	recommend("Pirates of the Caribbean: At World's End")
	```
	```
	Output:
	
	Pirates of the Caribbean: Dead Man's Chest
	Pirates of the Caribbean: On Stranger Tides
	Pirates of the Caribbean: The Curse of the Black Pearl
	20,000 Leagues Under the Sea
	Puss in Boots
	```