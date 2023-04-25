from sentence_transformers import  SentenceTransformer, util
import torch
import pickle
from sentence_transformers.cross_encoder import CrossEncoder

# Load fine-tuned models
model_bi_encoder = SentenceTransformer('output/training_bi_encoder_0409_best')
model_cross_encoder = CrossEncoder('output/cross_encoder_0409')

# read the information and embeddings for the movie data
with open("data/movie_cleaned", "rb") as f:
    titles,descriptions,movie_ids,keywords,genres,desc_for_embs,desc_embeddings,ratings = pickle.load(f)

# Your query : description about the movie (e.g : Documentaries about music)
query = ' Historical fiction about vikings and revenge'

# Encode the query with bi-encoder
query_embedding = model_bi_encoder.encode(query)
# Compute the cosine-similarity for all embedded movie description
cos_scores_bi = util.cos_sim(query_embedding, desc_embeddings)
# Find the top 50 movies as cross-encoder candidates
scores_check,indices_bi_encoder = torch.topk(cos_scores_bi, k=50)
# Get the movie description embeddings of top N movies
desc_for_embs_topN=[desc_for_embs[i] for i in indices_bi_encoder[0]]
# Get the query_desc combinations for cross-encoder
sentence_combinations = [[query, desc] for desc in desc_for_embs_topN]
# Calculate cross-encoder similarity scores
similarity_scores = model_cross_encoder.predict(sentence_combinations)
# Get the top 10 selected from cross-encoder
_,indices_cross_encoder_topn = torch.topk(torch.tensor(similarity_scores),k=10)
# indices of top 10
indices_cross_encoder = [indices_bi_encoder[0][i] for i in indices_cross_encoder_topn]

# ratings of the movie
ratings_final = [ratings[i]  for i in indices_cross_encoder]
# create a dictionary to sort by ratings
dictionary = dict(zip(indices_cross_encoder, ratings_final))
sorted_dict = dict(sorted(dictionary.items(), key= lambda item: item[1], reverse= True))

# recommend me the top 10
bi_encoder_result=indices_bi_encoder[0][0:15]

for i in dictionary.keys():
    print('Title: ' + titles[i])
    print('Rating: ' + ratings[i])
    print('Genre: ' + genres[i])
    print('Description: ' + descriptions[i])
    print('\n')

