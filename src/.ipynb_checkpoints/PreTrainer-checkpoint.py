import torch
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from DSA import *
import matplotlib.pyplot as plt
import torch.optim as optim
import math

class PreTrainer():

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
        
    def train_to_get_best_lr(self, train_loader):
            self.model.train()
            
            def trainDSA(new_lr):
                print(new_lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr[0]
                    
                loss_list = []
                true_labels, pred_labels = [], []
            
                for batch in tqdm(train_loader, desc='----- [Training] '):
                    texts, texts_mask, imgs, labels = batch
                    texts, texts_mask, imgs, labels = texts.to(self.device), texts_mask.to(self.device), imgs.to(self.device),labels.to(self.device)
                    pred, loss = self.model(texts, texts_mask, imgs, labels=labels)
                    # metric
                    loss_list.append(loss.item())
                    true_labels.extend(labels.tolist())
                    pred_labels.extend(pred.tolist())
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                accuracy = self.processor.metric(true_labels, pred_labels)
                return round(sum(loss_list) / len(loss_list), 5)
            # 参数设置
            particleNum = 2
            dimension = 1
            iterTimes = 3
            targetFunc = trainDSA
            xran = (3e-7, 3e-5)

            # 初始化DSA算法
            dsa = DSA(particleNum, dimension, iterTimes, targetFunc, xran)
            best_lr, best_position, gbestList = dsa.run()

            print("Best learning rate:", best_lr)
            print("Best position:", best_position)
            print("Gbest list:", gbestList)
            def plotFitness(ls, title=""):
                plt.title("Fitness Evolution Curve: " + title)
                plt.xlabel("iterations")
                plt.ylabel("fitness")
                x = [i for i in range(len(ls))]
                plt.plot(x, ls)


            def plotSemilogFitness(ls, title=""):
                plt.title("Semilog Fitness Evolution Curve: " + title)
                plt.xlabel("iterations")
                plt.ylabel("semilog fitness")
                plt.semilogy(ls)
            plotFitness(gbestList,"Fitness")
            plotSemilogFitness(gbestList,"SemilogFitness")
            return best_position[0]