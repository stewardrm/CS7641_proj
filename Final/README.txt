
The MovieLens data used for this report may be downloaded at the following link: https://files.grouplens.org/datasets/movielens/ml-20m.zip

recommender.py represents the code for the first module "Bi-encoder 1 + TF-IDF" and intantiated with the movie_lens_data class.  The code is intergrated into the other 
modules but may also be run standalone.  The movie_lens_data_class accepts parameters for either the fine-tuned "bert uncased L-2 H-128 A-2" (model 1) and “bert-base-uncased.
Note that due to the size of the second model the program may take up to 30 minutes to tokenize.  For convenience this object may be downloaded from 
https://drive.google.com/drive/folders/1URg0iNq_DcNLL5FBaTvyn573lJfI1wsH?usp=share_link.
     Once the movie_lens_data object is tokenized with self.bert_tokenize two methods are available to return movie recommendations
        1. get_recommendations_by_title(self, selected_title, num_recommendations=5):  Note the formot of selected_title must conform the "title" in movies data frame to 		     include year for example "Toy Story (1995)".
        2. get_recommendations_by_keyword_bert_tfidf(self, user_keywords, num_recommendations=5, bert_weight=0.5, tfidf_weight=0.5, w_ratings=True)
        See recommender.py comments for more details

fine_tuner_code folder:  This is the code used to fine tune the bert uncased L-2 H-128 A-2. Before running this code the MovieLens data set must be copied into the same
   directory as fine_tuner.py

fine_tuned folder:  This folder contains the results from fine_tuner.py and is what recommender.py uses when "model 1" is selected.

Evaluator folder:
	Based on paper “Evaluating Recommendation Systems” by Guy Shani and Asela Gunawardana, in order to evaluate content based recommender algorithms ofﬂine, it is necessary 	to simulate the behavior of users that interact with a recommendation system. Offline experimentation is a type of experimentation that uses historical data to simulate 	how users would interact with a recommendation system. This type of experimentation is attractive because it is quick, cheap, and can be used to test a wide range of 	algorithms. However, offline experiments can only answer a limited set of questions. For example, they can be used to compare the prediction power of different 	algorithms, but they cannot be used to directly measure the impact of a recommendation system on user behavior.
	It is important to use relevant and unbiased data for offline evaluation of recommender systems, in order to effectively compare algorithms before deployment. Pre-	filtering data by excluding users or items with low counts would introduce bias. Randomly sampling users and items can be a better method for reducing data, and 	correcting biases in the data may require techniques such as reweighing data.
	In this study, two evaluation functions have been added to the recommender.py:
	Function get_evaluation_title, takes in a query movie title and a list of movie title suggestions recommended by get_recommendations_by_title for an input query movie 	title and calculates precision score for recommending these movies to a subset of users based on their past movie ratings. The function first classifies users into groups 	based on the number of movies they have rated and assigns higher weights to smaller groups by adjusting the weights based on the number of users in each group. It then 	randomly samples a subset (2000) of these users and filters users and their movie list to only include the selected subset. It then calculates the precision score for the 	selected subset of users based on whether they have rated the selected movie and the suggested movies, and whether they have given high ratings to the selected movie. The 	precision score is calculated as the number of correctly recommended movies divided by the total number of recommended movies.
	The function returns the precision score is calculated as the average precision across all selected subset of users in the dictionary.
	The function get_evaluation_keyword is doing the same as function get_evaluation_title but takes in just a list of movie title suggestions recommended by 	get_recommendations_by_keyword_bert_tfidf for an input keyword and calculates precision score for recommending these movies to a subset of users based on their past movie 	ratings.

 
Due to gradescope space limitations some ancilliary files were ommitted from the final submission.  For all code and further examples, please refer to the GitHub repository
our team used to collobarate together at the link below (https://github.com/stewardrm/CS7643_proj/tree/main/Final). Included in this repository that is not included in 
the gradescope submission are the three fine-tuned models used in recommender:


