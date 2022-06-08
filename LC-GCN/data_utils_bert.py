# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
# from transformers import BertTokenizer
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import DataLoader, random_split


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.42B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


class Tokenizer4Bert:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return sequence
#         return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

# class Tokenizer(object):
#     def __init__(self, word2idx=None):
#         if word2idx is None:
#             self.word2idx = {}
#             self.idx2word = {}
#             self.idx = 0
#             self.word2idx['<pad>'] = self.idx
#             self.idx2word[self.idx] = '<pad>'
#             self.idx += 1
#             self.word2idx['<unk>'] = self.idx
#             self.idx2word[self.idx] = '<unk>'
#             self.idx += 1
#         else:
#             self.word2idx = word2idx
#             self.idx2word = {v:k for k,v in word2idx.items()}

#     def fit_on_text(self, text):
#         text = text.lower()
#         words = text.split()
#         for word in words:
#             if word not in self.word2idx:
#                 self.word2idx[word] = self.idx
#                 self.idx2word[self.idx] = word
#                 self.idx += 1

#     def text_to_sequence(self, text):
#         text = text.lower()
#         words = text.split()
#         unknownidx = 1
#         sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
#         if len(sequence) == 0:
#             sequence = [0]
#         return sequence


class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        fin = open(fname+'.tree', 'rb')
        idx2tree = pickle.load(fin)
        fin.close()


        all_data = []
        for i in range(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            
            context = text_left + " " + aspect + " " + text_right
#             print(context)
            context = context.strip()
            context_wo_aspect = text_left + " " + text_right
            context_wo_aspect = context_wo_aspect.strip()
            
            text_indices = tokenizer.text_to_sequence(context)
#             print(len(text_indices))
            left_indices = tokenizer.text_to_sequence(text_left)
            
            aspect_indices = tokenizer.text_to_sequence(aspect)
#             left_len = np.sum(left_indices != 0)
#             aspect_len = np.sum(aspect_indices != 0)
            left_len = len(left_indices)
            aspect_len = len(aspect_indices)
#             print(aspect_len)
            polarity = int(polarity) + 1
#             text_len = np.sum(text_indices != 0)
            text_len = len(text_indices)
#             print(text_len)
            concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + context + ' [SEP] ' + aspect + " [SEP]")
#             print(concat_bert_indices)
            concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
#             print(concat_segments_indices)
            #concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)
            
            left_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " [SEP]")
            dependency_graph = idx2graph[i]
#             print(dependency_graph)
            dependency_tree = idx2tree[i]
#             print(dependency_tree)
            data = {
                'concat_bert_indices': concat_bert_indices,#
                'concat_segments_indices': concat_segments_indices,#
                'text_indices': text_indices,#
                'left_indices': left_indices,#
                'aspect_indices': aspect_indices,
                #'context_indices': context_indices,
                'dependency_graph': dependency_graph,
                'dependency_tree': dependency_tree,
                'polarity': polarity,
                }
            all_data.append(data)
#             break
        return all_data

    def __init__(self, opt, dataset='rest14', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
        }
#         text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
#         if os.path.exists(dataset+'_word2idx.pkl'):
#             print("loading {0} tokenizer...".format(dataset))
#             with open(dataset+'_word2idx.pkl', 'rb') as f:
#                  word2idx = pickle.load(f)
#                  tokenizer = Tokenizer(word2idx=word2idx)
#         else:
#             tokenizer = Tokenizer()
#             tokenizer.fit_on_text(text)
#             with open(dataset+'_word2idx.pkl', 'wb') as f:
#                  pickle.dump(tokenizer.word2idx, f)
#         self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        tokenizer = Tokenizer4Bert()
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:
            valset_len = int(len(self.train_data) * opt.valset_ratio)
            self.train_data, self.val_data = random_split(self.train_data, (len(self.train_data)-valset_len, valset_len))
        else:
            self.val_data = self.test_data
    