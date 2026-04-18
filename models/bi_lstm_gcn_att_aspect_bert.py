# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention

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

class BI_LSTM_GCN_ATT_ASPECT_BERT(nn.Module):
    """
    A model integrating Graph Convolutional Networks (GCN) with BERT and Aspect Embedding
    for sentiment classification. The model uses BERT for sentence and aspect encoding
    and GCN layers to capture the graph structure.
    
    :param bert: Pre-trained BERT model.
    :param opt: Configuration object containing model hyperparameters such as BERT dimension, hidden dimension.
    """
    def __init__(self, bert, opt):
        super(BI_LSTM_GCN_ATT_ASPECT_BERT, self).__init__()
        # Load pre-trained BERT model
        self.bert = bert
        
        # Define Bi_LSTM layers
        self.bilstm = DynamicLSTM(opt.bert_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Graph convolution layers
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Attention layer
        self.attention = Attention(2 * opt.hidden_dim, score_function='bi_linear')

        # Fully connected layer for sentiment classification
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass through the model. The input consists of BERT token indices, segment ids,
        aspect indices, and the dependency graph for graph convolution.
        
        :param inputs: Tuple containing (text_bert_indices, bert_segments_ids, aspect_indices, dependency_graph).
        :return: Sentiment prediction scores.
        """
        text_bert_indices, text_indices, bert_segments_ids, aspect_indices, dependency_graph = inputs
        
        # Encode text and aspect using BERT
        text_bert, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        aspect_bert, _ = self.bert(aspect_indices, token_type_ids=torch.zeros_like(aspect_indices), output_all_encoded_layers=False)
        
        # Combine text and aspect representations
        combined = text_bert + aspect_bert
        
        # Calculate sequence lengths
        x_len = torch.sum(text_indices != 0, dim=-1)
        
        # Bi-LSTM layer with variable-length sequences
        lstm_out, (h_n, _) = self.bilstm(combined, x_len)
        
        seq_len = lstm_out.size(1)
        dependency_graph = dependency_graph[:, :seq_len, :seq_len]
        
        # Apply GCN layers
        gcn_output = F.relu(self.gc1(lstm_out, dependency_graph))
        gcn_output = F.relu(self.gc2(gcn_output, dependency_graph))
        
        # Apply attention over GCN outputs
        attn_output, attn_weights = self.attention(gcn_output, gcn_output)
        
        # Apply pooling over the sequence (after attention)
        pooled_features = torch.mean(attn_output, dim=1)
        
        # Apply dropout for regularization
        dropped_features = self.dropout(pooled_features)
        
        # Fully connected layer for sentiment prediction
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output
