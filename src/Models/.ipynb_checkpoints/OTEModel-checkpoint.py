import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet50
import torch.nn.functional as F
import timm
from .mamba.models_insect import build_vssm_models_ as mamba_model
from .DWT import DWT_2D

class wad_module(nn.Module):
    def __init__(self, wavename='haar'):
        super(wad_module, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)
        self.softmax = nn.Softmax2d()

    @staticmethod
    def get_module_name():
        return "wad"

    def forward(self, input):
        LL, LH, HL, _ = self.dwt(input)
        output = LL

        x_high = self.softmax(torch.add(LH, HL))
        AttMap = torch.mul(output, x_high)
        output = torch.add(output, AttMap)
        return output
    
class CrossModalAlignment(nn.Module):
    def __init__(self, region_dim, word_dim, hidden_dim):
        super(CrossModalAlignment, self).__init__()
        self.hidden_dim = hidden_dim
        self.region_proj = nn.Linear(region_dim, hidden_dim)
        self.word_proj = nn.Linear(word_dim, hidden_dim)

    def forward(self, image_features, text_features):
        # Project the image and text features to the hidden dimension
        image_proj = self.region_proj(image_features)  # [batch_size, hidden_dim]
        text_proj = self.word_proj(text_features)  # [batch_size, hidden_dim]

        # Reshape to allow batch matrix multiplication
        image_proj = image_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        text_proj = text_proj.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Compute the affinity matrix
        affinity_matrix = torch.bmm(image_proj, text_proj.transpose(1, 2))  # [batch_size, 1, 1]
        affinity_matrix = F.softmax(affinity_matrix / (self.hidden_dim ** 0.5), dim=-1)

        interactive_text_features = torch.bmm(affinity_matrix, text_proj)  # [batch_size, 1, hidden_dim]
        interactive_text_features = interactive_text_features.squeeze(1)  # [batch_size, hidden_dim]

        return interactive_text_features, image_proj.squeeze(1)  # Return the projected image features

class CrossModalGating(nn.Module):
    def __init__(self, hidden_dim):
        super(CrossModalGating, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_proj, interactive_text_features):
        gate_values = self.sigmoid(torch.sum(image_proj * interactive_text_features, dim=-1))
        fused_features = gate_values.unsqueeze(-1) * interactive_text_features + image_proj
        return fused_features

class TextModel(nn.Module):

    def __init__(self, config):
        super(TextModel, self).__init__()

        self.bert = AutoModel.from_pretrained(config.bert_name)
        self.trans = nn.Sequential(
            nn.Dropout(config.bert_dropout),
            nn.Linear(self.bert.config.hidden_size, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        ) 
        
        # 是否进行fine-tune
        for param in self.bert.parameters():
            if config.fixed_text_model_params:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def forward(self, bert_inputs, masks, token_type_ids=None):
        assert bert_inputs.shape == masks.shape, 'error! bert_inputs and masks must have same shape!'
        bert_out = self.bert(input_ids=bert_inputs, token_type_ids=token_type_ids, attention_mask=masks)
        pooler_out = bert_out['pooler_output']
        
        return self.trans(pooler_out)
    
class ImageModel(nn.Module):

    def __init__(self, config):
        super(ImageModel, self).__init__()
        self.mamba_model = mamba_model(cfg="vssm_tiny")
        in_features = self.mamba_model.classifier.head.in_features
        self.mamba_model.classifier.head = nn.Sequential(
            nn.Dropout(config.resnet_dropout),
            nn.Linear(in_features, config.middle_hidden_size),
            nn.ReLU(inplace=True)
        )
        
        # 实例化WAD模块
        self.wad = wad_module()

    def forward(self, imgs):
        imgs = self.wad(imgs)
        feature = self.mamba_model(imgs)
        return feature

class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        # text
        self.text_model = TextModel(config)
        # image
        self.img_model = ImageModel(config)
        # attention
        self.attention = nn.TransformerEncoderLayer(
            d_model=config.middle_hidden_size * 2,
            nhead=config.attention_nhead, 
            dropout=config.attention_dropout
        )
        # 全连接分类器
        self.classifier = nn.Sequential(
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.middle_hidden_size * 2, config.out_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(config.fuse_dropout),
            nn.Linear(config.out_hidden_size, config.num_labels)
        )
        self.loss_func = nn.CrossEntropyLoss()
        
        image_feature_dim = config.middle_hidden_size
        text_feature_dim = config.middle_hidden_size
        hidden_dim = config.out_hidden_size
        
        self.cross_modal_alignment = CrossModalAlignment(image_feature_dim, hidden_dim, hidden_dim)
        self.cross_modal_gating = CrossModalGating(hidden_dim)
        self.gru = nn.GRU(text_feature_dim, hidden_dim, batch_first=True)
        
        # Feedforward layers for combining features
        self.image_ff = nn.Linear(image_feature_dim, hidden_dim)
        self.text_ff = nn.Linear(hidden_dim, hidden_dim)
        self.gating_ff = nn.Linear(hidden_dim, hidden_dim)
        self.combine_ff = nn.Linear(hidden_dim * 3, hidden_dim)
        
    def forward(self, texts, texts_mask, imgs, labels=None):
        text_features = self.text_model(texts, texts_mask)
        image_features = self.img_model(imgs)
    
        # Pass text features through GRU
        text_features = text_features.unsqueeze(1)  # Adding sequence dimension
        gru_out, _ = self.gru(text_features)
        text_features = gru_out[:, -1, :]  # Use the last hidden state from GRU
        interactive_text_features, image_proj = self.cross_modal_alignment(image_features, text_features)
        fused_features = self.cross_modal_gating(image_proj, interactive_text_features)

        # Combine features through feedforward layers
        image_features_combined = self.image_ff(image_features)
        text_features_combined = self.text_ff(text_features)
        gating_features_combined = self.gating_ff(fused_features)

        combined_features = torch.cat((image_features_combined, text_features_combined, gating_features_combined), dim=1)
        combined_features = self.combine_ff(combined_features)

        attention_out = self.attention(combined_features)

        # attention_out = self.attention(torch.cat(
        #     [text_features.unsqueeze(0), image_features.unsqueeze(0)],
        # dim=2)).squeeze()

        prob_vec = self.classifier(attention_out)
        if len(prob_vec.shape) == 1:
            prob_vec = prob_vec.unsqueeze(0)
        pred_labels = torch.argmax(prob_vec, dim=1)

        if labels is not None:
            loss = self.loss_func(prob_vec, labels)
            return pred_labels, loss
        else:
            return pred_labels