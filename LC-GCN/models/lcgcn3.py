# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
import numpy as np


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj, tree):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(tree, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class LC_GCN(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(LC_GCN, self).__init__()
        self.opt = opt
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.text_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.gc1 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        self.gc2 = GraphConvolution(opt.hidden_dim, opt.hidden_dim)
        #self.gc3 = GraphConvolution(2*opt.hidden_dim, 2*opt.hidden_dim)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def mask(self, x, aspect_double_idx):
        batch_size, seq_len = x.shape[0], x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        mask = [[] for i in range(batch_size)]
        for i in range(batch_size):
            for j in range(aspect_double_idx[i,0]):
                mask[i].append(0)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                mask[i].append(1)
            for j in range(aspect_double_idx[i,1]+1, seq_len):
                mask[i].append(0)
        mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2).to(self.opt.device)
        return mask*x
    
    def feature_dynamic_weighted(self, text_local_indices, aspect_indices):
        max_seq_len = text_local_indices.shape[1]
        texts = text_local_indices.cpu().numpy()
        asps = aspect_indices.cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), max_seq_len, self.opt.embed_dim),
                                          dtype=np.float32)
        for text_i, asp_i in zip(range(len(texts)), range(len(asps))):
            asp_len = np.count_nonzero(asps[asp_i]) - 2
            try:
                asp_begin = np.argwhere(texts[text_i] == asps[asp_i][1])[0][0]
                asp_avg_index = (asp_begin * 2 + asp_len) / 2
            except:
                continue
            distances = np.zeros(np.count_nonzero(texts[text_i]), dtype=np.float32)
            for i in range(1, np.count_nonzero(texts[text_i])-1):
                if abs(i - asp_avg_index) + asp_len / 2 > self.opt.SRD:
                    #distances[i] = 1 / (abs(i - asp_avg_index) + asp_len/2 - self.opt.SRD)#-1次方
                    #distances[i] = (abs(i - asp_avg_index) + asp_len/2 - self.opt.SRD) ** (-2)#-2次方
                    #distances[i] = (abs(i - asp_avg_index) + asp_len/2 - self.opt.SRD) ** (-3)#-2次方
                    #distances[i] = (1 / 2) ** (abs(i - asp_avg_index) + asp_len/2 - self.opt.SRD) #1/2为底
                    #distances[i] = (1 / 3) ** (abs(i - asp_avg_index) + asp_len/2 - self.opt.SRD) #1/3为底
                    #distances[i] = 0 #CDM
                    #distances[i] = 1 #全
                    distances[i] = 1 - (abs(i - asp_avg_index)+asp_len/2- self.opt.SRD)/np.count_nonzero(texts[text_i])#1/n
                    
                else:
                    distances[i] = 1
                    
            for i in range(len(distances)):
                masked_text_raw_indices[text_i][i] = masked_text_raw_indices[text_i][i] * distances[i]
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.opt.device)
    
    def alpha(self, mask, gcn_out):
        alpha_mat = torch.matmul(mask, gcn_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2).permute(0, 2, 1)
        return alpha * gcn_out

    def forward(self, inputs):
        text_indices, aspect_indices, left_indices, adj, tree = inputs
        text_len = torch.sum(text_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
        text = self.embed(text_indices)
        text = self.text_embed_dropout(text)
#         weighted_text_local_features = self.feature_dynamic_weighted(text_indices, aspect_indices)#LSTM之前加上降权来代替dropout
#         weighted_out = torch.mul(text, weighted_text_local_features)#GCN之前使用降权
        text_out, (_, _) = self.text_lstm(text, text_len)
        #text_out_drop = self.text_embed_dropout(text_out)#lstm之后加上dropout
        
#         weighted_text_local_features = self.feature_dynamic_weighted(text_indices, aspect_indices)
#         weighted_out = torch.mul(text_out, weighted_text_local_features)#GCN之前使用降权
#         x = F.relu(self.gc1(weighted_out, adj, tree))
#         #weighted_out = torch.mul(x, weighted_text_local_features)
#         x = F.relu(self.gc2(x, adj, tree))
#         #weighted_out = torch.mul(x, weighted_text_local_features)
#         #x = F.relu(self.gc3(x, adj, tree))
        
#         直接从lstm得到的特征向量中提取方面词的特征
        aspect_mask = self.mask(text_out, aspect_double_idx)
        #在每层gcn使用相似度赋予权重
        weighted_out = self.alpha(aspect_mask, text_out)
        x = F.relu(self.gc1(weighted_out, adj, tree))
#         weighted_out = self.alpha(aspect_mask, x)
        x = F.relu(self.gc2(x, adj, tree))
        
        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        x = torch.matmul(alpha, text_out).squeeze(1) # batch_size x 2*hidden_dim
        output = self.fc(x)
        return output