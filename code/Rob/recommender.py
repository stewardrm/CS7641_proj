import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer, AutoModel #not used
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset


class movie_lens_data():
    def __init__(self, movies, ratings, genome_scores,genome_tags,tags,truncate=True):
        #loading the movielens csv into pandas data frames
        self.data1=pd.read_csv(movies, sep=',', header=0)
        self.data2=pd.read_csv(ratings, sep=',', header=0)
        self.data3=pd.read_csv(genome_scores,sep=',',header=0)
        self.data4=pd.read_csv(genome_tags,sep=',',header=0)
        self.data5=pd.read_csv(tags,sep=',',header=0)

        #organizing the data into named columns.  Variables are named after the movielens csv file names
        self.movies = self.data1[['movieId', 'title', 'genres']]
        self.ratings = self.data2[['userId', 'movieId', 'rating', 'timestamp']]
        self.genome_scores=self.data3[['movieId','tagId','relevance']]
        self.genome_tags=self.data4[['tagId','tag']]
        self.tags=self.data5[['userId','movieId','tag','timestamp']]

        #bert tokenizer and AutoModel
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        if truncate==True:
            self.movies=self.movies.head(100) #truncating here for faster testing

        self.titles = self.movies['title'].apply(lambda x: x.split(' (')[0])  # remove the year from the title
        self.titles=self.titles.apply(lambda x: x+" ")
        generes = self.movies['genres']  # using the genre as the movie description
        generes=generes.str.replace('|', ' ')
        self.combined=list(self.titles+generes)
        self.embeddings=None


    def bert_tokenize(self,batch_size=32):

        inputs = self.bert_tokenizer(self.combined, padding=True, truncation=True, return_tensors='pt')  #this seems to be the standard options
        df_length = len(self.combined)
        embeddings_list = []
        for i in range(0, df_length, batch_size):
          inputs = self.bert_tokenizer(self.combined[i:i+batch_size], padding=True, truncation=True, return_tensors='pt')
          with torch.no_grad(): # we are not training the model, but rather using the model to make predictions hence no_grad saves memory
            outputs = self.bert_model(**inputs) #the double-asterisk (**) operator describes how the inputs are unpacked to the self.bert_model function
            title_embeddings = outputs.last_hidden_state[:len(inputs), 0, :]
            desc_embeddings = outputs.last_hidden_state[len(inputs):, 0, :]
            self.embeddings = torch.cat((title_embeddings, desc_embeddings), dim=0)
            embeddings_list.append(self.embeddings)
        self.embeddings = torch.cat(embeddings_list, dim=0)


    def get_recommendations(self, selected_title, num_recommendations=5):
        # preprocess the query movie title and description
        query_title = selected_title.split(' (')[0]
        query_title=query_title+" "
        movies=self.movies
        query_desc = movies[movies['title'] == selected_title]['genres'].values[0]

        # tokenize and encode the query movie title and description with BERT
        query_input = self.bert_tokenizer([query_title + ' ' + query_desc], padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            query_embedding = self.bert_model(**query_input).last_hidden_state[:, 0, :]

        # compute the cosine similarity between the query movie and all other movies
        cosine_similarities = torch.nn.functional.cosine_similarity(self.embeddings, query_embedding)

        # get the indices of the top N recommendations, excluding any movies with the same title as the query movie
        query_index = self.titles.tolist().index(query_title)
        top_indices = cosine_similarities.argsort(descending=True)[1:]
        top_indices = [i for i in top_indices if i != query_index][:num_recommendations]
        # return the top N movie recommendations
        return movies.loc[top_indices]['title'].values.tolist()
