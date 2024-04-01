import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, c_in, c_out_mod,c_out_sig):
        super(CNN, self).__init__()

        self.embedding = nn.Sequential(
            nn.Conv1d(c_in, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool1d(2))
        self.mod_head = nn.Sequential(
            nn.Conv1d(16, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(c_out_mod))     
        self.sig_head = nn.Sequential(
            nn.Conv1d(16, 16, 3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LazyLinear(c_out_sig))  

    def forward(self, x):
        embedding_output = self.embedding(x)
        embedding_output = F.normalize(embedding_output)
        pred_mod = self.mod_head(embedding_output)
        pred_sig = self.sig_head(embedding_output)
        return pred_mod, pred_sig