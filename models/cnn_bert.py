# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CNN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(CNN_BERT, self).__init__()
        self.bert = bert
        self.conv1 = nn.Conv2d(1, opt.num_filters, (opt.kernel_size, opt.bert_dim)) 
        self.conv2 = nn.Conv2d(1, opt.num_filters, (opt.kernel_size, opt.bert_dim)) 
        self.fc = nn.Linear(opt.num_filters * 2, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.dropout_rate)
        
    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs
        encoder_layer, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        
        x = encoder_layer.unsqueeze(1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = nn.functional.relu(x1).squeeze(3)
        x2 = nn.functional.relu(x2).squeeze(3)
        x1 = nn.functional.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = nn.functional.max_pool1d(x2, x2.size(2)).squeeze(2)
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
