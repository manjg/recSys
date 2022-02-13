### Hybrid model based on contents and collaborative filtering

This Movie recommendation system recommends movies to users based on the dataset found [here](https://www.kaggle.com/rounakbanik/the-movies-dataset/data)

* movies_metadata.csv: The main Movies Metadata file. Contains information on 45,000 movies featured in the Full MovieLens dataset. Features include posters, backdrops, budget, revenue, release dates, languages, production countries and companies.
* keywords.csv: Contains the movie plot keywords for our MovieLens movies. Available in the form of a stringified JSON Object.
* credits.csv: Consists of Cast and Crew Information for all our movies. Available in the form of a stringified JSON Object.
* links.csv: The file that contains the TMDB and IMDB IDs of all the movies featured in the Full MovieLens dataset.
* links_small.csv: Contains the TMDB and IMDB IDs of a small subset of 9,000 movies of the Full Dataset.
* ratings_small.csv: The subset of 100,000 ratings from 700 users on 9,000 movies.
* ratings.csv: The full ratings dataset.
* Besides having the usual user-interaction data, this dataset also has some textual metadata (either in the `metadata.csv` file or in the `keywords.csv` file).

The provided `requirements.txt` file contains all the libraries required to run the tool. The tool can be run using:

`python recsys.py --in-folder <path-to-data> --out-folder <path-to-model-destination>` , where
	* `<path-to-data>` corresponds to the data containing the csv files
	* `<path-to-model-destination>` corresponds to a folder where the trained model will be serialised to
	
The code in `recsys.py`:
* generate a training / eval / test split (and do any necessary data pre-processing)
* train a model
* print evaluation metrics
* save it to a destination

The model gives movie suggestions to a particular user based on the similer contents as given movie and estimated ratings that it had internally calculated for that user.

The model uses the small subset of 9000 movies of the Full MovieLens dataset as per links_small.csv file due to resource limitaions.

The contents based model uses item metadata, such as genre, director, description, actors, language etc. for movies, and recommend the similer movies as the given item based on the cosine similarity.

Two different models are made for different contens based on:
TF-IDF vectoriser: movie tagline and overview (description)

CountVectorizer(): movie cast, crew, keywords, genre and language (metadata), because we do not want to down-weight the presence of an actor/director if he or she has acted or directed in relatively more movies. keyword are stemmed before feeding to CountVectorizer()

The models above are improved based on the specific (default 40%) vote_count quantile (i.e. movies with more votes than the 40% of the movies) are selected with top imdb ratings calculated as ((v/(v+m) * R) + (m/(m+v) * C)) where
v is the number of votes for the movie
m is the minimum votes required to be listed in the chart
R is the average rating of the movie
C is the mean vote across the whole report

The content based model is not really personel. It is only capable of suggesting movies which are close to a certain movie. That is, it is not capable of capturing tastes and providing recommendations across genres.

To improve upon the limitation above collabirative filtering is used to make recommendations. Collaborative model doesn't care what the movie is (or what it contains). It is based on the idea that users similar to me can be used to predict how much I will like a particular movie those users have liked but I have not. It works purely on the basis of an assigned movie ID and tries to predict ratings based on how the other users have predicted the movie.

Two collaborative models using Surprise Library based on singular value decomposition SVD and KNNWithMeans are made to predict the movie rating based on a user profile without knowing the past behaviour of the user. 

The hybrid model recommends movies based on either 'metadata' or 'description' and then refining the content based recommendation as per predicted movies ratings(using either SVD or KNNWithMeans) for the user.

Different combination of the models can be put through A/B testing to decide best recommendations.

### proposed improvements 
This is a basic model utilizing the content and collaborative filtering. However, as the collabirative filtering is applied on the results of the content based results, the recommendations for two different users won't be very different. 
Various meyhod can be used to improve and test it. e.g. 
Seperating the content based movies and the collaborative filtering recommendations as two different options

In the content based model, instead of using stemming in the keywords processing, lemmatization can be used to keep the context of the keywords. Also encoding/decoding the output of the CounterVectorizer() and TF-IDF models using deep neural networks might improve the results.

In the collaborative filtering, a few (e.g. 5) similer user/movie ratings can be predicted using SVD and then all these features can be passed to train/test another regression model to predict the better ratings for recommendations.
