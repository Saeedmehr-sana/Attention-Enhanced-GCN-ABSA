# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(CNN, self).__init__()
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.conv1 = nn.Conv2d(1, opt.num_filters, (opt.kernel_size, opt.embed_dim)) 
        self.conv2 = nn.Conv2d(1, opt.num_filters, (opt.kernel_size, opt.embed_dim)) 
        self.fc = nn.Linear(opt.num_filters * 2, opt.polarities_dim) 
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        text_indices = inputs[0]
        x = self.embed(text_indices)
        x = x.unsqueeze(1)
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
