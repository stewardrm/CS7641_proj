from sentence_transformers import  SentenceTransformer
import csv
from datetime import datetime

path='data/movies.csv'

titles = []
descriptions = []
movie_ids = list()
keywords = []
genres = []
desc_for_embs = []
ratings = []

movie_id = 0
count = 0
with open(path, 'rt',  encoding='utf8') as file:
    next(file)
    reader = csv.reader(file)
    for row in reader:
       if row[3] == 'en' and row[5] != '' and float(row[5])>=5 \
               and row[16] != '' \
               and row[4] != '' and row[2] != '' \
               and row[7] != '' and  datetime.strptime(row[7], '%Y-%m-%d') >= datetime.strptime('1985-01-01', '%Y-%m-%d'):
          data=row
          description = row[4]
          title = row[1]
          rating = row[13]
          genre = row[2].replace('-',', ')
          keyword = row[16].replace('-',', ')
          production = row[6].replace('-', ', ')
          movie_ids.append(movie_id)
          titles.append(title)
          genres.append(genre)
          keywords.append(keyword)
          descriptions.append(description)
          desc_for_emb = 'Movie genre includes '+genre+ '. ' + description + ' Some of the movie keywords include '+ keyword+ '. The movie is produced by ' + production + '.'
          desc_for_embs.append(desc_for_emb)
          ratings.append(rating)
          movie_id += 1

model_bi_encoder = SentenceTransformer('output/training_bi_encoder_0409_best')
desc_embeddings = model_bi_encoder.encode(desc_for_embs,show_progress_bar =True)

movie_referrence=[titles,descriptions,movie_ids,keywords,genres,desc_for_embs,desc_embeddings,ratings]

import pickle
with open("data/movie_cleaned", "wb") as f:
   pickle.dump(movie_referrence, f)