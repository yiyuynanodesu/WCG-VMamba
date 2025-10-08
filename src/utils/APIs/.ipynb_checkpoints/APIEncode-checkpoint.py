'''
encode api: 将原始data数据转化成APIDataset所需要的数据
    tips:
        ! 必须调用labelvocab的add_label接口将标签加入labelvocab字典
'''

from tqdm import tqdm
from transformers import AutoTokenizer
from torchvision import transforms

from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def api_encode(data, config,mode):

    ''' 文本处理 BERT的tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(config.bert_name)

    ''' 图像处理 torchvision的transforms '''
    def get_resize(image_size):
        for i in range(20):
            if 2**i >= image_size:
                return 2**i
        return image_size
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
                transforms.CenterCrop(config.image_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    img_transform = data_transforms[mode]

    ''' 对读入的data进行预处理 '''
    encoded_texts, encoded_imgs, labels = [], [], []
    for line in tqdm(data, desc='----- [Encoding]'):
        img, text, label = line
        
        # 文本
        text.replace('#', '')
        tokens = tokenizer.tokenize('[CLS]' + text + '[SEP]')
        encoded_texts.append(tokenizer.convert_tokens_to_ids(tokens))

        # 图像
        encoded_imgs.append(img_transform(img))
            
        # 标签
        labels.append(label)

    return encoded_texts, encoded_imgs, labels

