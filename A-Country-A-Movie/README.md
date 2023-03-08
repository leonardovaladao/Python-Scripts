# A-Country-a-Movie
A small Python program that makes a World Map showing best movie for each country, ranked by IMDB Score, using Pandas and Bokeh.

## How it was done

First, a dataset was taken from [this link](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset) with information from various IMDB movies.

After filtering the dataset for the best movie from each country (based on IMDB Score), the OMDB API is called to get general information for each movie (as year of release, genre, poster image, etc). The data taken by the API is stored in the dataset.

A map was made with help of [this link](https://towardsdatascience.com/a-complete-guide-to-an-interactive-geographical-map-using-python-f4c5197e23e0). Taking the map objects from it to generate a World Map with Python, the previous dataset with the movie's information was merged with the countrie's codes to match each country with it's respective movie. Then, a simple HTML handled the information it was supposed to be shown.

By the end, the Python file saves the work done in an HTML page, to be shown in web browser.

## Visualization

The work can be visualized in [this link](https://leonardovaladao.github.io/Python-Scripts/A-Country-A-Movie/map.html).
