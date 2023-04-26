#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer
from transformers import BertConfig
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer,AutoModel #not used
from transformers import BertTokenizer,BertModel
from torch.utils.data import DataLoader,TensorDataset



class movie_lens_data():
    def __init__(self,movies,ratings,genome_scores,genome_tags,tags,truncate=True, model=1):
        #model 1: fine_tuned_tokenizer (small)
        #model 2: off the shelf bert_uncased model (large)
        self.model=model
        self.data1=pd.read_csv(movies,sep=',',header=0)
        self.data2=pd.read_csv(ratings,sep=',',header=0)
        self.data3=pd.read_csv(genome_scores,sep=',',header=0)
        self.data4=pd.read_csv(genome_tags,sep=',',header=0)
        self.data5=pd.read_csv(tags,sep=',',header=0)

        self.test1=None
        self.test2=None

        #organizing the data into named columns.  Variables are named after the movielens csv file names
        self.movies=self.data1[['movieId','title','genres']]
        self.ratings=self.data2[['userId','movieId','rating','timestamp']]
        self.genome_scores=self.data3[['movieId','tagId','relevance']]
        self.genome_tags=self.data4[['tagId','tag']]
        self.tags=self.data5[['userId','movieId','tag','timestamp']]


        #assign model
        #model 1 is the fine tuned google/bert_uncased_L-2_H-128_A-2 model
        #model 2 is the original bert-base-uncased un-tuned model that is much larger
        if self.model==1:
            self.bert_tokenizer = BertTokenizer.from_pretrained('fine_tuned/fine_tuned_tokenizer')
            config = BertConfig.from_pretrained('fine_tuned/fine_tuned_bert_config.json')
            self.bert_model = BertForSequenceClassification(config)
            self.bert_model.load_state_dict(torch.load('fine_tuned/fine_tuned_bert_model.pt', map_location=torch.device('cpu')))
            self.bert_model.eval()
        elif self.model==2:
            self.bert_tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model=BertModel.from_pretrained('bert-base-uncased')

        #used to only get the first 100 movies in the data set for faster testing
        if truncate==True:
            self.movies=self.movies.head(100) #truncating here for faster testing
            self.genome_scores=self.genome_scores[self.genome_scores['movieId'] <= 100]
            self.tags=self.tags[self.tags['movieId'] <= 100]

        #removing the year of the title
        self.titles=self.movies['title'].apply(lambda x: x.split(' (')[0])  # remove the year from the title

        self.titles=self.titles.apply(lambda x: x+" ")  #adding an extra space to get the formatting correct
        generes=self.movies['genres']  # using the genre as the movie description
        generes=generes.str.replace('|',' ')  #removing the "|" seperator
        if model==1:
             self.combined=list(generes)  #creating a string which has both the reformatted titles and genres
        elif model==2:
             self.combined=list(self.titles+generes)  #creating a string which has both the reformatted titles and genres

        self.embeddings=None #initialize the embeddings to be used for Title genres
        self.movie_keyword_embeddings=None #initialize movie keyword embedding

        self.collection_of_keywords=self.build_keyword_collection()  #references below function to build a collection of keywords from the tags csv
        self.tfidf_vectorizer=TfidfVectorizer()  #initialized the TfidVectorizer class
        self.tfidf_transform=self.tfidf_vectorizer.fit_transform(self.collection_of_keywords)   #transforms keywords into their statistical frequency significance

    def bert_tokenize(self, batch_size=32):
        # Tokenize the title/genres strings and the keyword_collection
        features = self.bert_tokenizer(self.combined, padding=True, truncation=True, return_tensors='pt')
        df_length = len(self.combined)
        embeddings_list = []
        for i in range(0, df_length, batch_size):
            features = self.bert_tokenizer(self.combined[i:i + batch_size], padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                if self.model==1:
                    depend_variables=self.bert_model.bert(**features) #the double-asterisk (**) operator describes how the features are unpacked to the self.bert_model function
                    title_embeddings=depend_variables.last_hidden_state[:len(features),0,:] #selects the embeddings for the tokens of the title for each sample in the batch
                    desc_embeddings=depend_variables.last_hidden_state[len(features):,0,:] #selects the embeddings for the description for each sample in the batch.
                    self.embeddings=torch.cat((title_embeddings,desc_embeddings),dim=0)

                elif self.model==2:
                    depend_variables=self.bert_model(**features) #the double-asterisk (**) operator describes how the features are unpacked to the self.bert_model function
                    title_embeddings=depend_variables.last_hidden_state[:len(features),0,:] #selects the embeddings for the tokens of the title for each sample in the batch
                    desc_embeddings=depend_variables.last_hidden_state[len(features):,0,:] #selects the embeddings for the description for each sample in the batch.
                    self.embeddings=torch.cat((title_embeddings,desc_embeddings),dim=0)
                embeddings_list.append(self.embeddings)
        self.embeddings = torch.cat(embeddings_list, dim=0)



        #Now do basically the same thing, except with the keyword_collection
        keyword_collection = self.collection_of_keywords
        features = self.bert_tokenizer(keyword_collection.tolist(), padding=True, truncation=True, return_tensors='pt')
        df_length = len(keyword_collection)
        embeddings_list = []
        for i in range(0, df_length, batch_size):
            batch_features = self.bert_tokenizer(keyword_collection[i:i + batch_size].tolist(), padding=True, truncation=True, return_tensors='pt', max_length=512)
            with torch.no_grad():
                if self.model==1:
                    bert_output = self.bert_model.bert(**batch_features)
                    last_hidden_state = bert_output.last_hidden_state
                    batch_embeddings = last_hidden_state[:, 0, :128]
                elif self.model==2:
                    batch_depend_variables=self.bert_model(**batch_features)
                    batch_embeddings=batch_depend_variables.last_hidden_state[:,0,:]

                embeddings_list.append(batch_embeddings)
        self.movie_keyword_embeddings = torch.cat(embeddings_list, dim=0)
        self.collection_of_keywords = keyword_collection



    def build_keyword_collection(self):
        #build movie_genres
        movie_genres = self.movies.set_index('movieId')['genres']
        movie_genres = movie_genres.str.split('|', expand=True)
        movie_genres = movie_genres.stack().reset_index(level=1, drop=True)
        movie_genres = movie_genres.reset_index(name='tag')


        # create a genome_tags df wher the relevance is greater than a preset amount.  .9 is currently set, but we should test other values
        temp1 = self.genome_scores.merge(self.genome_tags, on='tagId')
        temp2 = temp1.query("relevance > 0.9")
        movie_genome_tags = temp2[['movieId', 'tag']]
        movie_user_tags=self.tags[['movieId','tag']]

        #now combine the three above dataframes and clean them up
        final_keywords=pd.concat([movie_genres,movie_genome_tags,movie_user_tags],ignore_index=True,sort=False)
        final_keywords=final_keywords.drop_duplicates().reset_index(drop=True)  #remove duplicate tags
        final_keywords=final_keywords.dropna(subset=['tag']) #drop rows with NaN values

        #create a dict between movieid and all their processed keywords
        temp1=final_keywords.groupby('movieId')  #grab the column
        temp2 = temp1['tag'].apply(lambda tags: {tag.lower() for tag in tags})

        #add the movie titles to movie_keywords
        movie_titles = self.movies.set_index('movieId')['title'].str.lower().to_dict()
        movie_keywords = {movie_id: set([movie_titles[movie_id]] + list(tags))
                      for movie_id, tags in temp2.items()}

        #finally create a space-separated string of all the tags associated with each movie. These will be sorted by the 'movieId' index
        temp1=pd.Series(movie_keywords).apply(lambda x: ' '.join(x))
        keyword_collection=temp1.sort_index()

        return keyword_collection

    def get_recommendations_by_title(self, selected_title, num_recommendations=5):
        # Preprocess the query movie title and description
        query_title = selected_title.split(' (')[0]
        query_title = query_title + " "
        movies = self.movies
        movies_with_ratings = self.movies.merge(self.ratings[['movieId', 'rating']], on='movieId')

        query_desc = movies[movies['title'] == selected_title]['genres'].values[0]
        # Use bert to encode the query title and description using the Bert model
#        query_input = self.bert_tokenizer([query_title + query_desc], padding=True, truncation=True, return_tensors='pt')
        query_input=self.bert_tokenizer([query_title+' '+query_desc],padding=True,truncation=True,return_tensors='pt')
        with torch.no_grad():
            if self.model==1:
                bert_output = self.bert_model.bert(**query_input)
                last_hidden_state = bert_output.last_hidden_state
                #query_embedding = last_hidden_state[:, 0, :128]
                query_embedding = last_hidden_state[:, 0, :]
            elif self.model==2:
                 query_embedding=self.bert_model(**query_input).last_hidden_state[:,0,:]

        # Compute the cosine similarity between the query movie and all other movies
        cosine_similarities = torch.nn.functional.cosine_similarity(self.embeddings, query_embedding)
        #print(self.embeddings.shape)
        #print(cosine_similarities.shape)
        #print(query_title)
        # Don't include any movies with the same title as the query, get the indices of the top N recommendations.
        query_index = self.titles.tolist().index(query_title)  #this is the row of the query movie in the movies_df
        top_indices = cosine_similarities.argsort(descending=True)[1:]
        self.test1=top_indices
        self.test2=cosine_similarities
        top_indices = [i for i in top_indices if i != query_index][:num_recommendations]
        # Return the top N movie recommendations
        return movies.iloc[top_indices]['title'].values.tolist()


    def get_recommendations_by_keyword_bert_tfidf(self, user_keywords, num_recommendations=5, bert_weight=0.5, tfidf_weight=0.5, w_ratings=True):
        # Use BERT to encode the user keywords using the Bert model
        user_keywords = user_keywords.lower()
        user_keywords_input = self.bert_tokenizer([user_keywords], padding=True, truncation=True, return_tensors='pt', max_length=512)

        with torch.no_grad():
            if self.model==1:
                bert_output = self.bert_model.bert(**user_keywords_input)
                last_hidden_state = bert_output.last_hidden_state
                user_keywords_embedding = last_hidden_state[:, 0, :128]
            if self.model==2:
                user_keywords_embedding=self.bert_model(**user_keywords_input).last_hidden_state[:,0,:]

        # Calculate the cosine similarity for both BERT and TF-IDF and then combine them per the hyperparameters
        bert_cosine_similarities = torch.nn.functional.cosine_similarity(self.movie_keyword_embeddings, user_keywords_embedding)
        tfidf_cosine_similarities = cosine_similarity(self.tfidf_vectorizer.transform([user_keywords]), self.tfidf_transform)

        combined_cosine_similarities = bert_weight * bert_cosine_similarities + tfidf_weight * torch.tensor(tfidf_cosine_similarities[0], dtype=torch.float32)

        if w_ratings:
            movie_ratings = self.ratings.groupby('movieId')['rating'].mean()
            # Use movie ratings as a weight factor
            movie_ids = self.movies['movieId'].tolist()
            rating_weights = [movie_ratings.get(mid, 0.0) for mid in movie_ids]
            rating_weights = torch.tensor(rating_weights, dtype=torch.float32)
            combined_cosine_similarities = combined_cosine_similarities * rating_weights

        # Get the top num_recommendations
        top_indices = combined_cosine_similarities.argsort(descending=True)
        top_indices = top_indices.flatten()
        top_indices = top_indices[:num_recommendations]
        recommended_movie_ids = self.collection_of_keywords.index[top_indices]

        return self.movies[self.movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
