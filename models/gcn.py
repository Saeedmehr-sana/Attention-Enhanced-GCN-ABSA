# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

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

class GCN(nn.Module):
    """
    A Graph Convolutional Network (GCN) model for sentiment analysis.
    This model uses graph convolution layers to process text and dependency graph.

    :param embedding_matrix: Pre-trained word embeddings matrix.
    :param config: Configuration object containing model hyperparameters.
    """
    def __init__(self, embedding_matrix, opt):
        super(GCN, self).__init__()
        # Initialize embedding layer with pre-trained embeddings
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        # Define two Graph Convolution layers
        self.gc1 = GraphConvolution(opt.embed_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Fully connected layer to classify into sentiment polarities
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Optional dropout layer to prevent overfitting
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass through the GCN model.
        x = torch.mean(x, dim=1)
        :param input_data: Tuple containing text indices and the adjacency matrix of the graph.
        :return: Sentiment prediction for each input.
        """
        text_indices, dependency_graph = inputs
        
        # Embed the text input into word embeddings
        embedded_text = self.embed(text_indices)
        
        # Apply the first Graph Convolution layer
        features_after_gcn1 = F.relu(self.gc1(embedded_text, dependency_graph))
        
        # Apply the second Graph Convolution layer
        features_after_gcn2 = F.relu(self.gc2(features_after_gcn1, dependency_graph))
        
        # Apply mean pooling over the sequence dimension
        pooled_features = torch.mean(features_after_gcn2, dim=1)
        
        # Apply dropout for regularization
        dropped_features = self.dropout(pooled_features)
        
        # Pass the pooled and dropped features through the fully connected layer
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
        
