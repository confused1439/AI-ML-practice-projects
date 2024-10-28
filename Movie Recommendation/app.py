import streamlit as st
import pickle

with open('movies.pkl', 'rb') as f:
    movies = pickle.load(f)

with open('cosine_similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

def recommend(movie_title):
   movie_idx = movies[movies['title'] == movie_title].index[0]
   distances = similarity[movie_idx]
   movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
   
   recommended_movie_lst = []
   for i in movies_list:
      recommended_movie_lst.append(movies.iloc[i[0]].title)
    
   return recommended_movie_lst

st.title("Movie Recommendation")

option = st.selectbox(
    "Select a movies and I will recommend you some...",
    movies['title'],
    index=0,
    placeholder="Select contact method...",
)

# st.write("You selected:", recommend(option))
st.write(f"You selected: {option}")

# Get the list of recommended movies
recommended_movies = recommend(option)

# Display the recommended movies in a 3x3 grid layout
st.header("Recommended Movies:")
cols = st.columns(3)  # Create 3 columns for each row

# Loop through the recommended movies and place them in the grid
for idx, movie in enumerate(recommended_movies):
    with cols[idx % 3]:  # Distribute movies across columns
        st.write(movie)
