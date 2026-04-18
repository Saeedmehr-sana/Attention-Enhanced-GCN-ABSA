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

class GCN_ASPECT(nn.Module):
    """
    A Graph Convolutional Network (GCN) model designed for sentiment analysis,
    which incorporates aspects in the form of aspect indices along with the text
    and dependency graph for better representation.
    
    :param embedding_matrix: Pre-trained word embeddings matrix.
    :param opt: Configuration object containing model hyperparameters (e.g., embedding dimension, hidden dimension).
    """
    def __init__(self, embedding_matrix, opt):
        super(GCN_ASPECT, self).__init__()
        # Embedding layer initialized with pre-trained embeddings
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        # Graph convolution layers
        self.gc1 = GraphConvolution(opt.embed_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Fully connected layer to output sentiment predictions
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(opt.dropout_rate)
        
    def forward(self, inputs):
        """
        Forward pass through the GCN model with text and aspect inputs.
        
        :param inputs: Tuple containing (text_indices, aspect_indices, dependency_graph).
        :return: Sentiment prediction for each input pair.
        """
        text_indices, aspect_indices, dependency_graph = inputs
        
        # Embedding the text and aspect using pre-trained embeddings
        embedded_text = self.embed(text_indices)
        embedded_aspect = self.embed(aspect_indices)
        
        # Combine text and aspect embeddings (element-wise sum or concatenation)
        combined = embedded_text + embedded_aspect
        
        # Apply graph convolution layers
        x = F.relu(self.gc1(combined, dependency_graph))
        x = F.relu(self.gc2(x, dependency_graph))
        
        # Pooling the features over the sequence dimension
        pooled_features = torch.mean(x, dim=1)
        
        # Apply dropout for regularization
        dropped_features = self.dropout(pooled_features)
        
        # Sentiment prediction through fully connected layer
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
        
