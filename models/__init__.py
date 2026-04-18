# -*- coding: utf-8 -*-

# text
from models.cnn import CNN
from models.lstm import LSTM
from models.bi_lstm import Bi_LSTM
from models.bi_lstm_cnn import Bi_LSTM_CNN
from models.gcn import GCN
from models.bi_lstm_gcn import Bi_LSTM_GCN
from models.gcn_attention import GCN_ATTENTION

# aspect
from models.cnn_aspect import CNN_ASPECT
from models.lstm_aspect import LSTM_ASPECT
from models.bi_lstm_aspect import Bi_LSTM_ASPECT
from models.bi_lstm_cnn_aspect import Bi_LSTM_CNN_ASPECT
from models.gcn_aspect import GCN_ASPECT
from models.bi_lstm_gcn_aspect import Bi_LSTM_GCN_ASPECT

# bert - text
from models.lstm_bert import LSTM_BERT
from models.bi_lstm_bert import Bi_LSTM_BERT
from models.cnn_bert import CNN_BERT
from models.bi_lstm_cnn_bert import Bi_LSTM_CNN_BERT
from models.gcn_bert import GCN_BERT
from models.bi_lstm_gcn_bert import Bi_LSTM_GCN_BERT

# bert - aspect
from models.lstm_aspect_bert import LSTM_ASPECT_BERT
from models.bi_lstm_aspect_bert import Bi_LSTM_ASPECT_BERT
from models.cnn_aspect_bert import CNN_ASPECT_BERT
from models.bi_lstm_cnn_aspect_bert import Bi_LSTM_CNN_ASPECT_BERT
from models.gcn_aspect_bert import GCN_ASPECT_BERT
from models.bi_lstm_gcn_aspect_bert import Bi_LSTM_GCN_ASPECT_BERT

from models.bi_lstm_gcn_att_aspect_bert import BI_LSTM_GCN_ATT_ASPECT_BERT
from models.bi_lstm_gcn_sentic_aspect_bert import BI_LSTM_GCN_SENTIC_ASPECT_BERT
from models.bi_lstm_gcn_att_sentic_dep_aspect_bert import BI_LSTM_GCN_ATT_SENTIC_DEP_ASPECT_BERT
from models.bi_lstm_gcn_att_sentic_aspect_bert import BI_LSTM_GCN_ATT_SENTIC_ASPECT_BERT
from models.bi_lstm_gcn_att_sentic_aspect import BI_LSTM_GCN_ATT_SENTIC_ASPECT


# import pickle
# with open( "save_tokenizer.p", "rb" ) as f:
#     tokenizer1 = pickle.load(f)

# for seq in context_indices[0:3]:
#     sequence = seq.cpu().tolist()  # Convert each sequence to a list of indices
#     print(tokenizer1.sequence_to_text(sequence))

# tokenizer1.sequence_to_text(text_indices[0].cpu().tolist())
