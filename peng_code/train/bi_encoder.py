import math
from sentence_transformers import models, losses
from sentence_transformers import  SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
import gzip
import csv
from torch.utils.data import DataLoader
import json
import torch
import random

model_name = 'nreimers/MiniLM-L6-H384-uncased' # Base pre-trained model
train_batch_size = 16
max_seq_length = 128 # texts longer than this will be truncated
num_epochs = 2 # number of epchos to train

# Same the model with name
model_version_name = '0409_best'
model_save_path = 'output/training_bi_encoder_'+ model_version_name

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# train set
train_set_1_path = 'data/yahoo_answers_question_answer.jsonl.gz'
train_set_2_path = 'data/gooaq_pairs.jsonl.gz'
train_set_path = [train_set_1_path,train_set_2_path]

# eval set
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

# Prepare the training set
training_set = []
prob = 0.25
for path in train_set_path:
    with gzip.open(path, 'rt', encoding='utf8') as fIn:
        for row in fIn:
            rand_num = random.random()
            if rand_num < prob:
                data = InputExample(texts=json.loads(row))
                training_set.append(data)


train_dataloader = DataLoader(training_set, shuffle=True, batch_size=train_batch_size)

# Loss function
train_loss = losses.MultipleNegativesRankingLoss(model)

# Read STSbenchmark dataset and use it as development set
dev_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        if row['split'] == 'dev':
            score = float(row['score']) / 5.0
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
torch.cuda.empty_cache()
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          optimizer_params= {'lr': 1e-07},
          evaluation_steps=int(len(train_dataloader)*0.02),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )

model_save_path_final = 'output/training_bi_encoder_'+ model_version_name+ '_final'
model.save(model_save_path_final,'training_bi_encoder_'+ model_version_name)