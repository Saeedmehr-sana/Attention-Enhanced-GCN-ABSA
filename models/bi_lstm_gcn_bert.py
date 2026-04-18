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

class Bi_LSTM_GCN_BERT(nn.Module):
    """
    A model that integrates Graph Convolutional Networks (GCN) with BERT for sentiment analysis.
    BERT is used for generating word embeddings, and GCN layers capture the graph structure 
    for better aspect-based sentiment prediction.
    
    :param bert: Pre-trained BERT model.
    :param opt: Configuration object containing model hyperparameters such as BERT dimension, hidden dimension.
    """
    def __init__(self, bert, opt):
        super(Bi_LSTM_GCN_BERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = bert
        
        # Define Bi_LSTM layers
        self.bilstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Graph convolution layers
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Fully connected layer for sentiment classification
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass through the model. The input consists of BERT token indices, segment ids,
        and the dependency graph for graph convolution.
        
        :param inputs: Tuple containing (text_bert_indices, bert_segments_ids, dependency_graph).
        :return: Sentiment prediction for each input sequence.
        """
        text_bert_indices, text_indices, bert_segments_ids, dependency_graph = inputs
        
        # Get the output of BERT
        encoder_layer, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        
        # Calculate sequence lengths
        x_len = torch.sum(text_indices != 0, dim=-1)
        
        # Bi-LSTM layer with variable-length sequences
        lstm_out, (h_n, _) = self.bilstm(encoder_layer, x_len)
        
        seq_len = lstm_out.size(1)
        dependency_graph = dependency_graph[:, :seq_len, :seq_len]
        
        # Apply the graph convolutional layers
        x = F.relu(self.gc1(lstm_out, dependency_graph))
        x = F.relu(self.gc2(x, dependency_graph))
        
        # Pooling the features across the sequence length dimension
        pooled_features = torch.mean(x, dim=1)
        
        # Apply dropout to the pooled features
        dropped_features = self.dropout(pooled_features)
        
        # Pass the features through the fully connected layer for sentiment classification
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
        

