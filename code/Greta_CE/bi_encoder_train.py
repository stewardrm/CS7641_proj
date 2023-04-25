from sentence_transformers import SentenceTransformer, SentencesDataset, util
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
import os
import csv
import pickle
import time
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
import math
import gzip
from zipfile import ZipFile
import random
from sentence_transformers import SentencesDataset, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from transformers.models.auto.auto_factory import model_type_to_module_name
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BE_Model:
    def __init__(self,model_type = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_type)
        self.model_save_path='output/bi_encoder_w_quora_CE'
        self.dataset_path_train = './data/classification/train_pairs.tsv'
        self.dataset_path_dev = './data/classification/dev_pairs.tsv'
        self.train_samples=[]
        self.dev_samples = []
        self.train_batch_size = 16
        self.num_epochs = 4

    def train(self, CE_Out,train_sentence_pairs):

        if not os.path.exists(self.model_save_path):
            #Read Quora data
            with open(self.dataset_path_train, 'r', encoding='utf8') as fIn:
                reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in reader:
                    self.train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))

            #Concat CE results to Quora training dataset      
            cross_encoder_train_samples = list(InputExample(texts=[data[0], data[1]], label=score) for (data, score) in zip(train_sentence_pairs, CE_Out))
            self.train_samples+=cross_encoder_train_samples
            dev_samples = []
            with open(self.dataset_path_dev, 'r', encoding='utf8') as fIn:
                reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
                for row in reader:
                    self.dev_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))


            #Evaluator for trained CE
            # train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
            train_dataset = SentencesDataset(self.train_samples, model)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.train_batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model)
            evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name='Quora-dev')
            


            #Use 10% of train data 
            warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)

            # Train the bi-encoder model
            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=self.num_epochs,
                      evaluation_steps=1000,
                      warmup_steps=warmup_steps,
                      output_path=self.model_save_path,
                      )
            
        else:
          model = SentenceTransformer(self.model_save_path)