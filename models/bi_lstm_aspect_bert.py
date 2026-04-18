# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class Bi_LSTM_ASPECT_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(Bi_LSTM_ASPECT_BERT, self).__init__()
        self.bert = bert
        self.bilstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text_bert_indices, text_indices, aspect_indices, bert_segments_ids = inputs
        
        encoder_layer, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        aspect = self.bert(aspect_indices, token_type_ids=None, output_all_encoded_layers=False)[0]
        
        combined_x = torch.cat([encoder_layer, aspect], dim=1)
        
        x_len = torch.sum(text_indices != 0, dim=-1)
        _, (h_n, _) = self.bilstm(combined_x, x_len)
        out = self.fc(h_n[0])
        return out
