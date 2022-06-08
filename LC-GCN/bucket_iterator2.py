# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy as np

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

class BucketIterator(object):
    def __init__(self, data, batch_size, sort_key='concat_bert_indices', shuffle=True, sort=True):
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size]))
        return batches

    def pad_data(self, batch_data):
        batch_concat_bert_indices = []
        batch_concat_segments_indices = []
        batch_text_indices = []
        batch_left_indices = []
        batch_aspect_indices = []
        batch_dependency_graph = []
        batch_dependency_tree = []
        batch_polarity = []
        max_len = max([len(t[self.sort_key]) for t in batch_data])
        for item in batch_data:
            concat_bert_indices, concat_segments_indices, text_indices, left_indices, aspect_indices, dependency_graph, dependency_tree, polarity = \
            item['concat_bert_indices'], item['concat_segments_indices'], item['text_indices'], item['left_indices'],\
                item['aspect_indices'], item['dependency_graph'], item['dependency_tree'], item['polarity']
            
            batch_concat_bert_indices.append(pad_and_truncate(concat_bert_indices, max_len, padding='post', truncating='post'))
            batch_concat_segments_indices.append(pad_and_truncate(concat_segments_indices, max_len))
            batch_text_indices.append(pad_and_truncate(text_indices, max_len, padding='post', truncating='post'))
            batch_left_indices.append(pad_and_truncate(left_indices, max_len, padding='post', truncating='post'))
            batch_aspect_indices.append(pad_and_truncate(aspect_indices, max_len, padding='post', truncating='post'))
            batch_dependency_graph.append(np.pad(dependency_graph, \
                ((0,max_len-dependency_graph.shape[0]),(0,max_len-dependency_graph.shape[0])), 'constant'))
            batch_dependency_tree.append(np.pad(dependency_tree, \
                ((0,max_len-dependency_tree.shape[0]),(0,max_len-dependency_tree.shape[0])), 'constant'))
            batch_polarity.append(polarity)
        return { \
                'concat_bert_indices': torch.tensor(batch_concat_bert_indices), \
                'concat_segments_indices': torch.tensor(batch_concat_segments_indices), \
                'text_indices': torch.tensor(batch_text_indices), \
                'left_indices': torch.tensor(batch_left_indices), \
                'aspect_indices': torch.tensor(batch_aspect_indices), \
                'dependency_graph': torch.tensor(batch_dependency_graph), \
                'dependency_tree': torch.tensor(batch_dependency_tree), \
                'polarity': torch.tensor(batch_polarity), \
            }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]
