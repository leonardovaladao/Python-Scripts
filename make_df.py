import pandas as pd

# Import DataFrame
# Dataset taken from https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset
df = pd.read_csv('movie_metadata.csv')[['movie_title', 'movie_imdb_link', 'country', 'imdb_score']].dropna()

# Search best movie for each country based on imdb_score
def SearchBestMovies(df):
    movieslist = pd.DataFrame(columns=['movie_title', 'movie_imdb_link', 'country', 'imdb_score'])
    countriesList = df['country'].unique()
    for i in range(0, len(countriesList)):
        country = df['country'].unique()[i]
        row = df[df['country']==country].sort_values(by='imdb_score', ascending=False).head(1)
        movieslist = movieslist.append(row)
    movieslist.reset_index(inplace=True)
    movieslist.drop('index', axis=1,inplace=True)
    return movieslist

# Call function
movies = SearchBestMovies(df)

# Import codes
codes = pd.read_csv('country_codes.csv')
codes.drop('Unnamed: 0', axis=1, inplace=True)

# Merge movies and codes
movies = pd.merge(left=movies, right=codes, left_on='country', right_on='Entity', how='left').drop(axis=0, index=[10, 28, 36])
movies.drop('country', axis=1, inplace=True)
movies = movies.reset_index().drop('index', axis=1)

# Create imdb_id column
imdb_id = []
for i in range(0, len(movies['movie_imdb_link'])):
        imdb_id.append(movies['movie_imdb_link'][i][28:35])
movies['imdb_id'] = imdb_id

# Create columns to be inserted by OMDB API
movies['Year'] = ""
movies['Genre'] = ""
movies['Director'] = ""
movies['Plot'] = ""
movies['Language'] = ""
movies['Poster'] = ""
movies['Response'] = ""

# Calls movie's information from OMDB API
import requests
import json
def call_imdb(api_key):
    for i in range(0, len(movies['imdb_id'])):
        movie_id = movies['imdb_id'][i]
        req = json.loads(requests.get('http://www.omdbapi.com/?i=tt'+movie_id+'&apikey='+api_key).text)
        movies['Year'][i] = req['Year']
        movies['Genre'][i] = req['Genre']
        movies['Director'][i] = req['Director']
        movies['Plot'][i] = req['Plot']
        movies['Language'][i] = req['Language']
        movies['Poster'][i] = req['Poster']
        movies['Response'][i] = req['Response']
        
# Insert your OMDB API key here!
key = '' 
call_imdb(key)

# Checks if every OMDB request was sucessful
for i in movies['Response']:
    if i==False:
        raise Exception('Response 400')

print(movies)