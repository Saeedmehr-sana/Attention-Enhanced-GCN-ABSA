# # -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from layers.dynamic_rnn import DynamicLSTM


class Bi_LSTM_CNN_ASPECT_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(Bi_LSTM_CNN_ASPECT_BERT, self).__init__()
        self.bert = bert
        self.bilstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.conv1 = nn.Conv2d(1, opt.num_filters, (opt.kernel_size, opt.hidden_dim * 2)) 
        self.conv2 = nn.Conv2d(1, opt.num_filters, (opt.kernel_size, opt.hidden_dim * 2))
        self.fc = nn.Linear(opt.num_filters * 2, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        text_bert_indices, text_indices, aspect_indices, bert_segments_ids = inputs
        
        encoder_layer, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        aspect = self.bert(aspect_indices, token_type_ids=None, output_all_encoded_layers=False)[0]
        
        combined_x = torch.cat([encoder_layer, aspect], dim=1)
        
        x_len = torch.sum(text_indices != 0, dim=-1)
        lstm_out, (h_n, _) = self.bilstm(combined_x, x_len) 
        
        x = lstm_out.unsqueeze(1)
        x1 = self.conv1(x)
        x1 = nn.functional.relu(x1)
        x1 = x1.squeeze(3)
        x1 = nn.functional.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = self.conv2(x)
        x2 = nn.functional.relu(x2)
        x2 = x2.squeeze(3)
        x2 = nn.functional.max_pool1d(x2, x2.size(2)).squeeze(2)
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        out = self.fc(x)
        return out
