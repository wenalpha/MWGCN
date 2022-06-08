# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def local(text, len_aspect, position):
    SRD = 5
    if position - SRD < 0:
        begin = 0
    else:
        begin = position - SRD
    if position + len_aspect + SRD < len(text.split()):
        end = position + len_aspect + SRD
    else:
        end = len(text.split())
    local_context = [i for i in text.split()][begin : end]
    return ' '.join(local_context)

def dependency_adj_matrix(text):
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def dependency_adj_matrix2(text, lcf, local_position, matrix):
    tokens = nlp(text)
    text_list = text.split()
    seq_len = len(text_list)
    sub_matrix = np.zeros((seq_len, seq_len)).astype('float32')
    lcf_list = lcf.split()

    #处理lcf部分
    for token in tokens:
        #如果当前节点是lcf,只有在里面的时候才会更改值
        if str(token) in lcf_list:
            #如果当前节点是aspect
            for j in range(seq_len):
                #只有当对方也是lcf才会改变
                if text_list[j] in lcf_list:
                    sub_matrix[token.i][j] = 1
                else:
                    sub_matrix[token.i][j] = 1 / 2 * (1 / (abs(j - local_position) + 1) + 1 / (abs(token.i - local_position) + 1))
        else:
            for j in range(seq_len):
                sub_matrix[token.i][j] = 1 / 2 * (1 / (abs(j - local_position) + 1) + 1 / (abs(token.i - local_position) + 1))
    return matrix * sub_matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    idx2tree = {}
    fout = open(filename+'.graph', 'wb')
    fout2 = open(filename+'.tree', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        text = text_left + ' ' + aspect + ' ' + text_right
        local_position = len(text_left.split()) + len(aspect.split()) / 2
        lcf = local(text, len(aspect.split()), len(text_left.split()))
        adj_matrix = dependency_adj_matrix(text)
        matrix = dependency_adj_matrix2(text, lcf, local_position, adj_matrix)
        idx2graph[i] = matrix
        idx2tree[i] = adj_matrix
    pickle.dump(idx2graph, fout)        
    fout.close()
    pickle.dump(idx2tree, fout2)        
    fout2.close()

if __name__ == '__main__':
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')