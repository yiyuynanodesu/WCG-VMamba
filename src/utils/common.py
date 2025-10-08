'''
普通的常用工具

'''

import os
import json
import chardet
import torch
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pywt

# 读取数据，返回[(image, describe, label)]元组列表
def read_from_file(path,only=None):
    data = []
    df = pd.read_csv(path)
    print('----- [Loading]')
    for index, row in df.iterrows():
        image_path, describe, label = row['image'], row['describe'], row['label']

        if only == 'text': img = Image.new(mode='RGB', size=(224, 224), color=(0, 0, 0))
        else:
            img = Image.open(image_path).convert('RGB')
            img.load()
    
        if only == 'img': text = ''

        data.append((img, describe, label))

    return data

# 分离训练集和验证集
def train_val_split(data, val_size=0.2):
    return train_test_split(data, train_size=(1-val_size), test_size=val_size)

# 写入数据
def write_to_file(path, outputs):
    df = pd.read_csv(path)
    df['predict'] = outputs
    df.to_csv(path,index=False)

# 保存模型
def save_model(output_path, model_type, model):
    output_model_dir = os.path.join(output_path, model_type)
    if not os.path.exists(output_model_dir): os.makedirs(output_model_dir)    # 没有文件夹则创建
    model_to_save = model.module if hasattr(model, 'module') else model     # Only save the model it-self
    output_model_file = os.path.join(output_model_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)