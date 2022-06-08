# -*- coding: utf-8 -*-
# ------------------
# @Author: BinLiang
# @Mail: bin.liang@stu.hit.edu.cn
# ------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM


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

    def forward(self, text, adj):
        hidden = torch.matmul(text.float(), self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class INTERGCN_BERT(nn.Module):
    def __init__(self, bert, opt):
        super(INTERGCN_BERT, self).__init__()
        self.opt = opt
        self.bert = bert

        self.gc1 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc2 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc3 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        self.gc4 = GraphConvolution(opt.bert_dim, opt.bert_dim)
        
#         tensor1 = torch.tensor([[  101, 12136,  2001, 12501,  1010,  2317,  4511,  4010,  1010,  8808,
#                   1051, 18153,  2075,  7249,  1012,   102, 12136,   102]])
#         tensor2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
#         encoder_layer, pooled_output = self.bert(tensor1, token_type_ids = tensor2, output_all_encoded_layers=False)
#         print(encoder_layer)


        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.text_embed_dropout = nn.Dropout(0.3)

    def position_weight(self, x, aspect_double_idx, text_len, aspect_len):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        aspect_double_idx = aspect_double_idx.cpu().numpy()
        text_len = text_len.cpu().numpy()
        aspect_len = aspect_len.cpu().numpy()
        weight = [[] for i in range(batch_size)]
        for i in range(batch_size):
            context_len = text_len[i] - aspect_len[i]
            for j in range(aspect_double_idx[i,0]):
                weight[i].append(1-(aspect_double_idx[i,0]-j)/context_len)
            for j in range(aspect_double_idx[i,0], aspect_double_idx[i,1]+1):
                weight[i].append(0)
            for j in range(aspect_double_idx[i,1]+1, text_len[i]):
                weight[i].append(1-(j-aspect_double_idx[i,1])/context_len)
            for j in range(text_len[i], seq_len):
                weight[i].append(0)
        weight = torch.tensor(weight).unsqueeze(2).to(self.opt.device)
#         print('weight', weight.shape)
        return weight*x

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
        mask = torch.tensor(mask).unsqueeze(2).float().to(self.opt.device)
        return mask*x
    #计算相似度权重， 输入要比较的两个要比较相似度的特征向量返回形状为[batch_size, seq_len, dim]
    #mask表示提取到的方面次特征，gcn_out表示每层gcn得到的结果
#     def alpha(self, mask, gcn_out):
        

    def forward(self, inputs):
        
#         tensor1 = torch.tensor([[  101, 12136,  2001, 12501,  1010,  2317,  4511,  4010,  1010,  8808,
#                   1051, 18153,  2075,  7249,  1012,   102, 12136,   102]])
#         tensor2 = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
#         encoder_layer, pooled_output = self.bert(tensor1.to(self.opt.device), token_type_ids = tensor2.to(self.opt.device), output_all_encoded_layers=False)
#         print('now', encoder_layer)
        
        text_bert_indices, bert_segments_ids, aspect_indices, left_indices, text_indices, adj, d_adj = inputs
#         print(text_bert_indices.shape, bert_segments_ids.shape)
#         print('text_bert_indices', text_bert_indices.dtype)
#         print('bert_segments_ids', bert_segments_ids)
#         print('aspect_indices', aspect_indices)
#         print('left_indices', left_indices)
#         print('text_indices', text_indices)
#         print('adj', adj)
#         print('d_adj', d_adj)
        text_len = torch.sum(text_indices != 0, dim=-1)
#         print(text_len)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        left_len = torch.sum(left_indices != 0, dim=-1)
        aspect_double_idx = torch.cat([left_len.unsqueeze(1), (left_len+aspect_len-1).unsqueeze(1)], dim=1)
#         print(aspect_double_idx)

        encoder_layer, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
#         print(encoder_layer.shape)


        text_out = encoder_layer
#         print(text_out)
        weighted_text = self.position_weight(text_out, aspect_double_idx, text_len, aspect_len)
#         print(weighted_text)
#         print('weighted_text', weighted_text.shape)
        

        x = F.relu(self.gc1(self.position_weight(text_out, aspect_double_idx, text_len, aspect_len), adj))
        x = F.relu(self.gc2(self.position_weight(x, aspect_double_idx, text_len, aspect_len), adj))

        x_d = F.relu(self.gc3(self.position_weight(x, aspect_double_idx, text_len, aspect_len), d_adj))
        x_d = F.relu(self.gc4(self.position_weight(x_d, aspect_double_idx, text_len, aspect_len), d_adj))

        x += 0.2 * x_d

        x = self.mask(x, aspect_double_idx)
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)#[batch_size, 1, seq_len]
        alpha1 = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2).permute(0, 2, 1)
#         print('alpha', alpha.shape)
#         print(alpha)
#         print('alpha1', alpha1.shape)
#         print(alpha1)
        x = torch.matmul(alpha, text_out).squeeze(1)

        output = self.fc(x)
        return output
