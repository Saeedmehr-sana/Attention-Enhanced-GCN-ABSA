# The class `GCN_ATTENTION` implements a Graph Convolutional Network (GCN) model for sentiment
# analysis using graph convolution layers and attention mechanism.
# -*- coding: utf-8 -*-

import math
import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention, NoQueryAttention

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

class GCN_ATTENTION(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(GCN_ATTENTION, self).__init__()
        # Initialize embedding layer with pre-trained embeddings
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        # Define two Graph Convolution layers
        self.gc1 = GraphConvolution(opt.embed_dim, opt.hidden_dim)
        self.gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        
        # Fully connected layer to classify into sentiment polarities
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        
        # Dropout for embeddings
        self.text_embed_dropout = nn.Dropout(opt.dropout_rate)
        
        # # Optional dropout layer to prevent overfitting
        # self.dropout = nn.Dropout(opt.dropout_rate)
        
        # Attention mechanism
        self.self_att = Attention(opt.embed_dim, score_function='bi_linear')

    def forward(self, inputs):
        text_indices, dependency_graph = inputs
        
        # Embed the text input
        embedded_text = self.embed(text_indices)  # Shape: [batch_size, seq_len, embed_dim]
        
        # Apply dropout to embeddings
        text = self.text_embed_dropout(embedded_text)
        
        # Self-attention to adjust the dependency graph
        _, score = self.self_att(text, text)
        dependency_graph = torch.mul(dependency_graph, score)
        
        # Graph Convolution Layers
        features_after_gcn1 = F.relu(self.gc1(text, dependency_graph))  # Shape: [batch_size, seq_len, hidden_dim]
        features_after_gcn2 = F.relu(self.gc2(features_after_gcn1, dependency_graph))  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Apply mean pooling
        pooled_features = torch.mean(features_after_gcn2, dim=1)  # Shape: [batch_size, hidden_dim]
        
        # Attention mechanism over GCN outputs
        alpha_mat = torch.matmul(features_after_gcn2, text.transpose(1, 2))  # Shape: [batch_size, seq_len, seq_len]
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)  # Shape: [batch_size, 1, seq_len]
        x = torch.matmul(alpha, text).squeeze(1)  # Shape: [batch_size, embed_dim]
        
        # Combine pooled features with attention output
        combined_features = pooled_features + x  # Combine both representations
        
        # Sentiment classification
        sentiment_output = self.fc(combined_features)  # Shape: [batch_size, polarities_dim]
        
        # # Pass the pooled and dropped features through the fully connected layer
        # sentiment_output = self.fc(dropped_features)
        
        return sentiment_output


    # def forward(self, inputs):
        
    #     text_indices, dependency_graph = inputs
    #     embedded_text = self.embed(text_indices)
        
    #     #ADDED
    #     text = self.text_embed_dropout(embedded_text)
        
    #     _, score = self.self_att(text, text)
    #     dependency_graph = torch.mul(dependency_graph, score)
        
    #     features_after_gcn1 = F.relu(self.gc1(text, dependency_graph))
    #     features_after_gcn2 = F.relu(self.gc2(features_after_gcn1, dependency_graph))
    #     pooled_features = torch.mean(features_after_gcn2, dim=1)
    #     # dropped_features = self.dropout(pooled_features)
    #     x = features_after_gcn2
        
    #     #ADDED
    #     alpha_mat = torch.matmul(x, text.transpose(1, 2))
    #     alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2) 
    #     x = torch.matmul(alpha, text).squeeze(1) 
        
    #     sentiment_output = self.fc(dropped_features)
    #     return sentiment_output