# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class Bi_LSTM_GCN_ASPECT(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(Bi_LSTM_GCN_ASPECT, self).__init__()
        # Initialize embedding layer with pre-trained embeddings
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        # Define Bi_LSTM layers
        self.bilstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Define two Graph Convolution layers
        self.gc1 = GraphConvolution(2*opt.embed_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Fully connected layer to classify into sentiment polarities
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Optional dropout layer to prevent overfitting
        self.dropout = nn.Dropout(opt.dropout_rate)
        
        # self.print_cont = 0
        
    def forward(self, inputs):
        text_indices, aspect_indices, dependency_graph = inputs
        
        # Embedding the text and aspect using pre-trained embeddings
        embedded_text = self.embed(text_indices)
        embedded_aspect = self.embed(aspect_indices)
        
        # Combine text and aspect embeddings (element-wise sum or concatenation)
        combined = embedded_text + embedded_aspect
        
        # Calculate sequence lengths
        x_len = torch.sum(text_indices != 0, dim=-1)
        
        # Bi-LSTM layer with variable-length sequences
        lstm_out, (h_n, _) = self.bilstm(combined, x_len)
        
        # Apply graph convolution layers
        x = F.relu(self.gc1(lstm_out, dependency_graph))
        x = F.relu(self.gc2(x, dependency_graph))
        
        # Pooling the features over the sequence dimension
        pooled_features = torch.mean(x, dim=1)
        
        # Apply dropout for regularization
        dropped_features = self.dropout(pooled_features)
        
        # Sentiment prediction through fully connected layer
        sentiment_output = self.fc(dropped_features)
        # x = F.softmax(sentiment_output, dim=1)
        
        # if self.print_cont == 0:
        #     print(f'output : {x}')
        #     print(f'shape_output : {x.shape}')
        #     self.print_cont = 1
        
        return sentiment_output
        