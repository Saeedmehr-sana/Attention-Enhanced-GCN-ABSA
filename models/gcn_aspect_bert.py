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

class GCN_ASPECT_BERT(nn.Module):
    """
    A model integrating Graph Convolutional Networks (GCN) with BERT and Aspect Embedding
    for sentiment classification. The model uses BERT for sentence and aspect encoding
    and GCN layers to capture the graph structure.
    
    :param bert: Pre-trained BERT model.
    :param opt: Configuration object containing model hyperparameters such as BERT dimension, hidden dimension.
    """
    def __init__(self, bert, opt):
        super(GCN_ASPECT_BERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = bert
        
        # Graph convolution layers
        self.gc1 = GraphConvolution(opt.bert_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Fully connected layer for sentiment classification
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass through the model. The input consists of BERT token indices, segment ids,
        aspect indices, and the dependency graph for graph convolution.
        
        :param inputs: Tuple containing (text_bert_indices, bert_segments_ids, aspect_indices, dependency_graph).
        :return: Sentiment prediction scores.
        """
        text_bert_indices, bert_segments_ids, aspect_indices, dependency_graph = inputs
        
        # Encode text and aspect using BERT
        text_repr, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        aspect_repr, _ = self.bert(aspect_indices, token_type_ids=torch.zeros_like(aspect_indices), output_all_encoded_layers=False)
        
        # # Mean pooling for aspect representation
        # aspect_representation = torch.mean(aspect_layer, dim=1)  # Shape: [batch_size, bert_dim]
        # # Transform aspect representation
        # aspect_transformed = F.relu(self.aspect_fc(aspect_representation))  # Shape: [batch_size, bert_dim]
        # # Combine aspect representation with sentence representation
        # aspect_expanded = aspect_transformed.unsqueeze(1).expand_as(encoder_layer)  # Shape: [batch_size, seq_len, bert_dim]
        # combine = encoder_layer + aspect_expanded  # Shape: [batch_size, seq_len, bert_dim]
        
        # Combine text and aspect representations
        combined_repr = text_repr + aspect_repr
        
        # Apply GCN layers
        gcn_output = F.relu(self.gc1(combined_repr, dependency_graph))
        gcn_output = F.relu(self.gc2(gcn_output, dependency_graph))
        
        # Apply pooling over the sequence
        pooled_features = torch.mean(gcn_output, dim=1)
        
        # Apply dropout for regularization
        dropped_features = self.dropout(pooled_features)
        
        # Fully connected layer for sentiment prediction
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
