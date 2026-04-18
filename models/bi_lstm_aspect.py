# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class Bi_LSTM_ASPECT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Bi_LSTM_ASPECT, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.bilstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_indices = inputs[0]
        aspect_indices = inputs[1]
        text = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        combined_x = text + aspect
        # combined_x = torch.cat([text, aspect], dim=1)
        x_len = torch.sum(text_indices != 0, dim=-1)
        _, (h_n, _) = self.bilstm(combined_x, x_len)
        out = self.fc(h_n[0])
        return out
