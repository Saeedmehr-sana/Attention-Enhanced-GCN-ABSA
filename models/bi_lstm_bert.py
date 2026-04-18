# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class Bi_LSTM_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(Bi_LSTM_BERT, self).__init__()
        self.bert = bert
        self.bilstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
    
    def forward(self, inputs):
        text_bert_indices, text_indices, bert_segments_ids = inputs
        encoder_layer, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        x_len = torch.sum(text_indices != 0, dim=-1)
        _, (h_n, _) = self.bilstm(encoder_layer, x_len)
        out = self.fc(h_n[0])
        return out
