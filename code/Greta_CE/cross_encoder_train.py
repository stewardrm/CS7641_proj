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

class CE_Model:
    def __init__(self,model_type = 'nreimers/MiniLM-L6-H384-uncased'):
        self.model = CrossEncoder(model_type)
        self.model_save_path='output/training_quora'
        self.dataset_path_train = './data/classification/train_pairs.tsv'
        self.dataset_path_dev = './data/classification/dev_pairs.tsv'
        self.train_samples=[]
        self.dev_samples = []
        self.train_batch_size = 16
        self.num_epochs = 4
        self.test_sentences=[]
        self.test_labels = []
        self.dataset_path_test = './data/classification/test_pairs.tsv'

    def train(self):
        if not os.path.exists(self.model_save_path):
          #Train CrossEncoder with Quora Duplicate Detection task
          #Read Quora data
          with open(self.dataset_path_train, 'r', encoding='utf8') as fIn:
              reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
              for row in reader:
                  self.train_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))
                  self.train_samples.append(InputExample(texts=[row['question2'], row['question1']], label=int(row['is_duplicate'])))

          
          with open(self.dataset_path_dev, 'r', encoding='utf8') as fIn:
              reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
              for row in reader:
                  self.dev_samples.append(InputExample(texts=[row['question1'], row['question2']], label=int(row['is_duplicate'])))

          #Evaluator for trained CE
          train_dataloader = DataLoader(self.train_samples, shuffle=True, batch_size=self.train_batch_size)
          evaluator = CEBinaryClassificationEvaluator.from_input_examples(self.dev_samples)

          warmup_steps = math.ceil(len(train_dataloader) * self.num_epochs * 0.1)

          # Train the model
          self.model.fit(train_dataloader=train_dataloader,
                    evaluator=evaluator,
                    epochs=self.num_epochs,
                    evaluation_steps=5000,
                    warmup_steps=warmup_steps,
                    output_path=self.model_save_path)
        else:
            self.model = CrossEncoder(self.model_save_path)

    def evaluate(self):
        with open(self.dataset_path_test, encoding='utf8') as fIn:
            reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                self.test_sentences.append([row['question1'], row['question2']])
                self.test_labels.append(int(row['is_duplicate']))
        evaluator = CEBinaryClassificationEvaluator(self.test_sentences, self.test_labels)
        return evaluator(self.model)








