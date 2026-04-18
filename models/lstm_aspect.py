# -*- coding: utf-8 -*-

from layers.dynamic_rnn import DynamicLSTM
import torch
import torch.nn as nn


class LSTM_ASPECT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LSTM_ASPECT, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):

        text_indices = inputs[0]
        aspect_indices = inputs[1]
        text = self.embed(text_indices)
        aspect = self.embed(aspect_indices)
        combined_x = text + aspect
        
        x_len = torch.sum(text_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(combined_x, x_len)
        out = self.fc(h_n[0])

        return out


        # self.print_cont = 0
        
        # combined_x = torch.cat([text, aspect], dim=1)
        # combined_x = text_x + aspect_x
        # combined = torch.cat([text_indices, aspect_indices], dim=1)
        # combined_x = self.embed(combined)
        # if self.print_cont == 0:
        #     # print(f'input : {text_indices}')
        #     # print(f'text : {text_indices}')
        #     # print(f'aspect : {aspect_indices}')
        #     # print(f'combined_x : {combined_x}')
        #     print(f'shape_text : {text_indices.shape}')
        #     print(f'shape_aspect : {aspect_indices.shape}')
        #     print(f'shape_combined_x : {combined_x.shape}')
        #     self.print_cont = 1
        
        # if self.print_cont == 0:
        #     print(f'output : {out}')
        #     print(f'shape_output : {out.shape}')
        #     self.print_cont = 1