import math
from sentence_transformers import  SentenceTransformer, InputExample,util
import gzip
import csv
from torch.utils.data import DataLoader
import json
import random
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers.cross_encoder import CrossEncoder

# data_load
# train set
train_set_1_path = 'data/yahoo_answers_question_answer.jsonl.gz'
train_set_2_path = 'data/gooaq_pairs.jsonl.gz'
train_set_path = [train_set_1_path,train_set_2_path]

random.seed(1101)

# Prepare the training set
qid = [] #query id
query = [] #query
answer_shuffled = [] #answer that is shuffled
train_pair_samples = [] #accurate pairs
qid_cnt = 0 #qid counter
prob = 0.25 # % of all pairs being loaded

for path in train_set_path:
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        for row in fIn:
            rand_num = random.random()
            if rand_num < prob:
                data = json.loads(row)
                query_d = json.loads(row)[0]
                answer_d = json.loads(row)[1]
                query.append(query_d)
                answer_shuffled.append(answer_d)
                qid.append(qid_cnt)
                train_pair_samples.append(InputExample(texts=[query_d, answer_d], label=1.0)) # generate train data (accurate pair) for sentence transformer
                qid_cnt += 1

random.shuffle(answer_shuffled) # Shuffled the answer

### Get the bi-encoder score for every random pair, and use the score as data labels for training cross-encoder non-matching labels.

model = SentenceTransformer('output/training_bi_encoder_0409_best') # read the bi-encoder
cos_scores_shuffle = []
train_shuffle_samples = []
for id in qid :
    query_embedding=model.encode(query[id],convert_to_tensor=True)
    ans_embedding= model.encode(answer_shuffled[id],convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, ans_embedding)
    train_shuffle_samples.append(InputExample(texts=[query[id], answer_shuffled[id]], label=cos_scores))
    if id % 1000 == 0:
        print(str(id) + ' out of '+ str(len(qid)))

training_set = train_pair_samples + train_shuffle_samples

#import pickle
#with open("training_set_cross_encoder", "wb") as f:
#   pickle.dump(training_set, f)

import pickle
with open('training_set_cross_encoder', 'rb') as f:
    training_set = pickle.load(f)

# Read STSbenchmark dataset and use it as development set
dev_samples = []
# eval set
sts_dataset_path = 'data/stsbenchmark.tsv.gz'
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_batch_size = 16
num_epochs = 2
model_save_path = 'output/cross_encoder_0409'
train_dataloader = DataLoader(training_set, shuffle=True, batch_size=train_batch_size)
model_cs_encoder = CrossEncoder('nreimers/MiniLM-L6-H384-uncased', num_labels=1,max_length  = 128)

# We add an evaluator, which evaluates the performance during training
evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

# Train the model
import torch
torch.cuda.empty_cache()
model_cs_encoder.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          optimizer_params={'lr': 1e-06},
          evaluation_steps=int(len(train_dataloader) * 0.02),
          warmup_steps=warmup_steps,
          output_path=model_save_path)