import fire
import numpy as np
import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
import pickle
from surprise import KNNWithMeans, SVD, Reader, Dataset
from surprise import accuracy


#clean movie_id function
def clean_id(x):
	try:
		return int(x)
	except:
		return np.nan

#get only the words
def filter_keywords(x,s):
	words = []
	for i in x:
		if i in s:
			words.append(i)
	return words

# keywords stemming and keep the keywords that appear more than once
def preprocess_keywords(movie_df: pd.DataFrame) -> pd.DataFrame :
	kw = movie_df.apply(lambda x: pd.Series(x['keywords'],dtype=object),axis=1).stack().reset_index(level=1, drop=True)
	kw.name = 'keyword'
	kw = kw.value_counts()
	kw = kw[kw>1]    				# get the keywords that appear more than once
	stemmer = SnowballStemmer('english')
	movie_df['keywords'] = movie_df['keywords'].apply(lambda x:[w for w in x if w in kw] )
	movie_df['keywords'] = movie_df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])   #get the root word by stemming
	return movie_df

#calculate imdb weighted ratings
def calc_weighted_rating(x,m,C):
	v = x['vote_count']
	R = x['vote_average']
	return (v/(v+m) * R) + (m/(m+v) * C)
	
def save_model(model, mdl_file: str):
	"""
	Serialise the model to an output folder 
	"""
	print("\nSaving the model to: ",mdl_file)
	with open(mdl_file,'wb') as sf:
		pickle.dump(model,sf)
	


def evaluate_model(model, test_data, label, reader):
	"""
	Evaluate your model against the test data.
	"""
	test_df = Dataset.load_from_df(test_data[['userId','movieId','rating']], reader)
	test_output = model.test(test_df.construct_testset(raw_testset=test_df.raw_ratings))
	
	print("\nRMSE -",label, accuracy.rmse(test_output, verbose = False))
	print("MAE -", label, accuracy.mae(test_output, verbose=False))



def split_data(data):
	"""
	Generate data splits
	"""
	train_set, test_set = train_test_split(data, test_size = 0.3)
	return train_set, test_set

## content based models
def metadata_count_vec(mov_df: pd.DataFrame) -> csr_matrix:
	count_vec = CountVectorizer(stop_words='english')
	count_vec.fit(mov_df['metadata'])
	return count_vec

def description_tfidf(mov_df: pd.DataFrame) -> csr_matrix:
	tfidf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
	tfidf.fit(mov_df['description'])
	return tfidf

## collabortive models

def create_KNN_SVD():
	# SVD
	algo_SVD = SVD()
	# KNN
	similarity = {
			"name": "cosine",
			"user_based": False,  # item-based similarity
	}
	algo_KNN = KNNWithMeans(sim_options = similarity)
	return algo_SVD, algo_KNN

def train_algo(algo, train_set):
	# load df into Surprise Reader object
	reader = Reader(rating_scale = (0,5))
	train_df = Dataset.load_from_df(train_set[['userId','movieId','rating']], reader)
	algo.fit(train_df.build_full_trainset())
	return algo, reader

# Hybrid model: to recommend movies bases on the textual data and users ratings
def movie_recommendations(user_id, title, movies_df, sim_matrix, movie_map, algo, cut_off=0.40):
	idx = movie_map[title.lower()]
	sim_scores = list(enumerate(sim_matrix[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	sim_scores = sim_scores[1:30]				##select top 25 movies based on similarity score
	movie_indices = [i[0] for i in sim_scores]
	## select movies based on the top imdb ratings
	movies = movies_df.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'movieId']]
	vote_counts = movies[movies['vote_count'].notnull()]['vote_count']  
	vote_averages = movies[movies['vote_average'].notnull()]['vote_average'] 
	C = vote_averages.mean()
	m = vote_counts.quantile(cut_off)  #minimum votes required to be listed in the recommendation, default as 40% quantile
	qualified = movies.loc[((movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())),:].copy()
	qualified.loc[:,'vote_count'] = qualified['vote_count']  
	qualified.loc[:,'vote_average'] = qualified['vote_average'] 
	qualified.loc[:,'wr'] = qualified.apply(lambda x:calc_weighted_rating(x,m,C), axis=1)			# calculate imdb rating
	qualified = qualified.sort_values('wr', ascending=False).head(20)

	qualified.loc[:,'rating'] = qualified['movieId'].apply(lambda x: algo.predict(user_id, x).est)
	qualified = qualified.sort_values('rating', ascending=False).reset_index(drop=True)

	return qualified[['title','rating']].head(10)

def load_data(in_folder: str):
	"""
	Load the csv file and join them in a format you can use
	"""
	ratings = pd.read_csv(in_folder + "ratings_small.csv")
	links = pd.read_csv(in_folder + "links_small.csv")
	movies_md = pd.read_csv(in_folder + "movies_metadata.csv", dtype='object')
	credits = pd.read_csv(in_folder + "credits.csv")
	#keywords like jealousy, friendship, rivalry etc that belongs to particular movies are also part of the metadata.
	#we will grab keywords from keywords.csv
	keywords = pd.read_csv(in_folder + "keywords.csv")

	#only released movies
	movies_md = movies_md[movies_md.status=='Released']
	
	movies_md['imdb_id'] = movies_md['imdb_id'].apply(lambda x: str(x)[2:].lstrip("0"))
	movies_md['imdb_id'] = movies_md['imdb_id'].apply(clean_id)
	movies_md = movies_md.dropna(subset=['imdb_id']).reset_index(drop=True)
	movies_md['imdb_id'] = movies_md['imdb_id'].astype('int')
	#select movies and ratings that are in links
	movies_md = movies_md.merge(links[['movieId','imdbId']],left_on='imdb_id',right_on='imdbId')
	ratings = ratings.drop(columns='timestamp')
	ratings = ratings[ratings.movieId.isin(links.movieId)]
	#ratings=ratings.merge(links[['movieId','imdbId']],on='movieId')

	movies_md['year'] = pd.to_datetime(movies_md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)

	movies_md['id'] = movies_md['id'].apply(clean_id)
	keywords['id'] = keywords['id'].apply(clean_id)
	credits['id'] = credits['id'].apply(clean_id)
	ratings['movieId'] = ratings['movieId'].apply(clean_id)

	movies_md['vote_count'] = movies_md['vote_count'].astype('int')
	movies_md['vote_average'] = movies_md['vote_average'].astype('float')
	movies_md = movies_md.dropna(subset=['id','vote_count','vote_average']).reset_index(drop=True)
	

	# converting all the 'id' into integer
	movies_md['id'] = movies_md['id'].astype('int')
	keywords['id'] = keywords['id'].astype('int')
	credits['id'] = credits['id'].astype('int')
	
	#merging the 3 dataframes to get all the required data on 1 datafarame movies
	movie_cols = ['id','imdb_id','movieId','title', 'year', 'vote_count', 'vote_average', 'popularity','original_language','genres','tagline','overview']
	movies_df = movies_md[movie_cols].merge(credits, on='id')
	movies_df = movies_df.merge(keywords, on='id')

	#changing the 4 columns into python objects ( list of dictionaries here) for metadata processing
	movies_df['genres'] = movies_df['genres'].apply(literal_eval)
	movies_df['cast'] = movies_df['cast'].apply(literal_eval)
	movies_df['crew'] = movies_df['crew'].apply(literal_eval)
	movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)

	#grab the names of all the genres attached to each movie
	movies_df['genres'] = movies_df['genres'].apply(lambda x: [i['name'].lower() for i in x])
	#grab the name of the director from all the crew members
	#we will only use directors from the crew column for our purpose
	movies_df['crew'] = movies_df['crew'].apply(lambda x: [i['name'].lower() for i in x if i['job']=='Director'])
	#grab the cast and keywords from the list of dictionaries of those columns
	movies_df['cast'] = movies_df['cast'].apply(lambda x: [i['name'].lower() for i in x])
	movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i['name'].lower() for i in x])
	movies_df = movies_df.drop_duplicates().reset_index(drop=True)

	#keep only the keywords that appear more than once
	movies_df = preprocess_keywords(movies_df) 

	#taking maximum 3 cast/genre/keywords for each movie
	movies_df['genres'] = movies_df['genres'].apply(lambda x: x[:3] if len(x)>3 else x)
	movies_df['cast'] = movies_df['cast'].apply(lambda x: x[:3] if len(x)>3 else x)
	movies_df['keywords'] = movies_df['keywords'].apply(lambda x: x[:3] if len(x)>3 else x)

	#removing spaces
	movies_df['cast'] = movies_df['cast'].apply(lambda x: [i.replace(' ','') for i in x])
	movies_df['crew'] = movies_df['crew'].apply(lambda x: [i.replace(' ','') for i in x])
	movies_df['keywords'] = movies_df['keywords'].apply(lambda x: [i.replace(' ','') for i in x])
	movies_df['genres'] = movies_df['genres'].apply(lambda x: [i.replace(' ','') for i in x])
	#combine the metadata
	movies_df['metadata'] = movies_df.apply(lambda x : ' '.join(x['genres']) + ' ' + ' '.join(x['original_language']) + ' ' 
					+ ' '.join(x['cast']) + ' ' + ' '.join(x['crew']) + ' ' + ' '.join(x['keywords']), axis = 1)

	movies_df['tagline'] = movies_df['tagline'].fillna('')
	movies_df['description'] = movies_df['overview'] + movies_df['tagline']
	movies_df['description'] = movies_df['description'].fillna('')

	return movies_df, ratings


def train(in_folder: str, out_folder: str) -> None:
	"""
	Consume the data from the input folder to generate the model 
	and serialise it to the out_folder
	"""
	mov_df, rat_df = load_data(in_folder)

	train_set, test_set = split_data(mov_df)
	train_set.reset_index(drop=True,inplace=True)
	test_set.reset_index(drop=True,inplace=True)

	## content based models
	## metadata counter-vectorization model
	mdl_count_vec = metadata_count_vec(train_set)
	mdl_file = out_folder+ "mdl_metadata.pkl"
	save_model(mdl_count_vec, mdl_file)

	#description tf-idf model
	mdl_tfidf = description_tfidf(train_set)
	mdl_file = out_folder+ "mdl_dscr.pkl"
	save_model(mdl_tfidf, mdl_file)

	## collaborative models
	algo_SVD, algo_KNN = create_KNN_SVD() 

	train_rat, test_rat = split_data(rat_df)

	## train and save the models
	algo_SVD, reader_svd = train_algo(algo_SVD, train_rat)
	mdl_file = out_folder + "algo_svd.pkl"
	save_model(algo_SVD,mdl_file)

	algo_KNN, reader_knn = train_algo(algo_KNN, train_rat)
	mdl_file = out_folder + "algo_knn.pkl"
	save_model(algo_KNN,mdl_file)

	## evaluate models
	evaluate_model(algo_SVD, test_rat, 'algo_SVD', reader_svd)
	evaluate_model(algo_KNN, test_rat, 'algo_KNN', reader_knn)

	## make some predictions for the test set

	#calculate metadata similarity matrix for the test set
	count_vec_matrix = mdl_count_vec.transform(test_set['metadata'])
	metadata_sim = cosine_similarity(count_vec_matrix)

	#calculate description similarity matrix for the test set
	tfidf_matrix = mdl_tfidf.transform(test_set['description'])
	description_sim = cosine_similarity(tfidf_matrix)

	#movies index mapping
	movie_map = pd.Series(test_set.index,index = test_set['title'].str.lower())
	##movie_map.to_csv(out_folder+'movie_map.csv')
	user_id = 670
	movie = test_set['title'].iloc[2]
	contents = ['metadata','description']

	print("\nLet's predict the recommendation for user: {0} based on the movie: {1}: ".format(user_id, movie))
	for sim_matrix, label in zip([metadata_sim, description_sim], contents):
		for algo in [algo_SVD, algo_KNN]:
			algo_name = str(algo).split(' ')[0].split('.')[-1]
			print("\nRecomendations based on: {0} using algorithm: {1}".format(label,algo_name))
			recommendations = movie_recommendations(user_id, movie, test_set, sim_matrix, movie_map, algo, cut_off=0.40)
			print(recommendations)



if __name__ == '__main__':

	fire.Fire(train)
 
