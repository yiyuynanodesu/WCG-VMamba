import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from DSA import *
import torch.nn as nn

import torch.optim as optim
import math

from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter

class Trainer():

    def __init__(self, config, processor, model, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.config = config
        self.processor = processor
        self.model = model.to(device)
        self.device = device
       
        bert_params = set(self.model.text_model.bert.parameters())
        mamba_params = set(self.model.img_model.mamba_model.parameters())
        other_params = list(set(self.model.parameters()) - bert_params - mamba_params)
        no_decay = ['bias', 'LayerNorm.weight']
        params = [
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.bert_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.text_model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.bert_learning_rate, 'weight_decay': 0.0},
            {'params': [p for n, p in self.model.img_model.mamba_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': self.config.resnet_learning_rate, 'weight_decay': self.config.weight_decay},
            {'params': [p for n, p in self.model.img_model.mamba_model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': self.config.resnet_learning_rate, 'weight_decay': 0.0},
            {'params': other_params,
                'lr': self.config.learning_rate, 'weight_decay': self.config.weight_decay},
        ]  
        self.optimizer = AdamW(params, lr=config.learning_rate)

    def train(self, train_loader):
            self.model.train()
            
            loss_list = []
            true_labels, pred_labels = [], []
            
            for batch in tqdm(train_loader, desc='----- [Training] '):
                texts, texts_mask, imgs, labels = batch
                texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device),labels.to(self.device)
                prob_vec, loss = self.model(texts, texts_mask, imgs, labels=labels)
                if len(prob_vec.shape) == 1:
                    prob_vec = prob_vec.unsqueeze(0)
                pred = torch.argmax(prob_vec, dim=1)
                # metric
                loss_list.append(loss.item())
                true_labels.extend(labels.tolist())
                pred_labels.extend(pred.tolist())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss = round(sum(loss_list) / len(loss_list), 5)
            train_accuracy = self.processor.metric(true_labels, pred_labels)
            return train_loss

    def valid(self, val_loader):
        self.model.eval()

        val_loss = 0
        true_labels, pred_labels = [], []

        for batch in tqdm(val_loader, desc='\t ----- [Validing] '):
            texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device),labels.to(self.device)
            prob_vec = self.model(texts, texts_mask, imgs, labels=labels)
            if len(prob_vec.shape) == 1:
                prob_vec, loss = prob_vec.unsqueeze(0)
            pred = torch.argmax(prob_vec, dim=1)
            # metric
            val_loss += loss.item()
            true_labels.extend(labels.tolist())
            pred_labels.extend(pred.tolist())
            
        accuracy = self.processor.metric(true_labels, pred_labels)
        return val_loss / len(val_loader), accuracy
            
    def predict(self, test_loader):
        self.model.eval()
        pred_labels = []

        for batch in tqdm(test_loader, desc='----- [Predicting] '):
            texts, texts_mask, imgs, labels = batch
            texts, texts_mask, imgs = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device)
            prob_vec = self.model(texts, texts_mask, imgs)
            if len(prob_vec.shape) == 1:
                prob_vec = prob_vec.unsqueeze(0)
            pred = torch.argmax(prob_vec, dim=1)

            pred_labels.extend(pred.tolist())

        return pred_labels