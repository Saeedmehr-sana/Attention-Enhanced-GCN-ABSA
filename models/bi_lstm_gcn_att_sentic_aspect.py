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

class BI_LSTM_GCN_ATT_SENTIC_ASPECT(nn.Module):
    """
    A model integrating Graph Convolutional Networks (GCN) with BERT and Aspect Embedding
    for sentiment classification. The model uses BERT for sentence and aspect encoding
    and GCN layers to capture the graph structure.
    
    :param bert: Pre-trained BERT model.
    :param opt: Configuration object containing model hyperparameters such as BERT dimension, hidden dimension.
    """
    def __init__(self, embedding_matrix, opt):
        super(BI_LSTM_GCN_ATT_SENTIC_ASPECT, self).__init__()        
        # Initialize embedding layer with pre-trained embeddings
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        # Define Bi_LSTM layers
        self.bilstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Define two Graph Convolution layers
        self.gc1 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.gc2 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        
        # Attention layer
        self.attention1 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention2 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        self.attention3 = Attention(2 * opt.hidden_dim, score_function='bi_linear')
        
        # Fully connected layer to classify into sentiment polarities
        self.fc = nn.Linear(2*opt.hidden_dim, opt.polarities_dim)
        
        # Optional dropout layer to prevent overfitting
        self.dropout = nn.Dropout(opt.dropout_rate)

    def forward(self, inputs):
        """
        Forward pass through the model. The input consists of BERT token indices, segment ids,
        aspect indices, and the dependency graph for graph convolution.
        
        :param inputs: Tuple containing (text_bert_indices, bert_segments_ids, aspect_indices, dependency_graph).
        :return: Sentiment prediction scores.
        """
        
        text_indices, aspect_indices, sdat_graph = inputs
        
        # Embedding the text and aspect using pre-trained embeddings
        embedded_text = self.embed(text_indices)
        embedded_aspect = self.embed(aspect_indices)
        
        # Combine text and aspect embeddings (element-wise sum or concatenation)
        combined = embedded_text + embedded_aspect
        
        # Calculate sequence lengths
        x_len = torch.sum(text_indices != 0, dim=-1)
        
        # Bi-LSTM layer with variable-length sequences
        lstm_out, (h_n, _) = self.bilstm(combined, x_len)

        seq_len = lstm_out.size(1)
        sdat_graph = sdat_graph[:, :seq_len, :seq_len]
        
        # Apply GCN layers
        gcn_output = F.relu(self.gc1(lstm_out, sdat_graph))
        gcn_output = F.relu(self.gc2(gcn_output, sdat_graph))

        # Apply attention over GCN outputs
        attn_output, attn_weights = self.attention1(gcn_output, gcn_output)
        attn_output, attn_weights = self.attention2(attn_output, attn_output)
        attn_output, attn_weights = self.attention3(attn_output, attn_output)
        
        # Apply pooling over the sequence (after attention)
        pooled_features = torch.mean(attn_output, dim=1)
        
        # Apply dropout for regularization
        dropped_features = self.dropout(pooled_features)
        
        # Fully connected layer for sentiment prediction
        sentiment_output = self.fc(dropped_features)
        
        return sentiment_output