#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  :
"""tfidf 模型
"""
import argparse
import os

import gensim
import jieba
from Intelligent_QuaAndAns.Config.allconfig import *


class Tf_Idf_Model(object):
    def __init__(self, data, dictionary_filepath, stop_words_file, model_name='tfidf'):
        self.cwd = os.getcwd()
        self.data = data  # json格式的数据集
        self.titles = []  # 未分词的问题
        self.content = []  # 已分词去停用词的问题
        self.dictionary_filepath = dictionary_filepath
        self.stop_words_file = stop_words_file
        for i in range(len(data)):
            self.titles.append(data[i]["title"])
            self.content.append(data[i]["split_title"])
        self.content_list = [line.rstrip().split() for line in self.content]

        with open(self.stop_words_file, encoding='utf-8') as f:
            lines = f.readlines()
        self.stop_list = [x.strip() for x in lines]

        self.dictionary = self.bulid_dictionary()
        self.model_name = model_name
        self.idx, self.tfidf = self.build_model(self.model_name)

    @staticmethod
    def split_word(query, stop_list):
        words = jieba.cut(query)
        result = ' '.join([word for word in words if word not in stop_list])
        return result

    def bulid_dictionary(self):
        """
        得到一个基于数据集的字典，每个词的编号
        :return:
        """
        if os.path.exists(self.dictionary_filepath):
            dictionary = gensim.corpora.Dictionary.load(
                self.dictionary_filepath)
        else:
            content_list = [line.rstrip().split() for line in self.content]
            dictionary = gensim.corpora.Dictionary(content_list)
            dictionary.save(self.dictionary_filepath)
        return dictionary

    def build_model(self, model_name):
        corpus = [self.dictionary.doc2bow(line) for line in self.content_list] # [(1, 1), (5, 1), (6, 1), (7, 1)]
        num_features = max(self.dictionary.token2id.values())
        tfidf = gensim.models.TfidfModel(corpus)
        if self.model_name == 'lsi':
            model = gensim.models.LsiModel(corpus)
            idx = gensim.similarities.SparseMatrixSimilarity(
                model[corpus], num_features=num_features)
        else:
            idx = gensim.similarities.SparseMatrixSimilarity(
                tfidf[corpus], num_features=num_features) # 将语料库转换到tfidf,稀疏矩阵压缩成密集化的并对它进行索引（ tfidf[corpus] --> (token_id, token_tfidf)）
        return idx, tfidf

    def get_topn_sims(self, sentences, n=5):
        """
        输入问题找出对应的前n相似度的编号
        :param sentences: 输入问题
        :param n: 默认输出5条
        :return:
        """
        split_sent = self.split_word(sentences, self.stop_list).split()
        # 得到匹配结果1
        results_1 = {'title': sentences, 'split_title': split_sent}
        vec = self.dictionary.doc2bow(split_sent)
        sims = self.idx[self.tfidf[vec]]

        # print("sims:",sims[0:10])  #sims: [0.00267427 0.01338223 0.00098903 0.3847431  0.01875558 0.01224608 0.         0.00742276 0.00146245 0.0115432 ]
        similarities = list(enumerate(sims))
        # print ("similarities:", similarities[0:10]) #similarities: [(0, 0.002674266), (1, 0.01338223), (2, 0.0009890312), (3, 0.3847431), (4, 0.018755578), (5, 0.012246077), (6, 0.0), (7, 0.0074227587), (8, 0.0014624505), (9, 0.011543197)]
        dict_sims = dict(similarities)
        # print ("dict_sims:", dict_sims)  #{0:0.111,1:0.555,...}

        # sorted_sims = sorted(dict_sims.values(), reverse=True)
        sorted_sims = sorted(dict_sims.items(),
                             key=lambda item: item[1],
                             reverse=True)

        # [(60486, 0.8786309), (42967, 0.74683285), (58056, 0.7258308), (53468, 0.6781828), (42972, 0.6732913), (22799, 0.6719236), (48435, 0.66815436), (57952, 0.65864986), (21258, 0.650809),
        print("sorted_sims:", sorted_sims[0:10])
        topn_sims = []
        sum = n
        i = 0
        while (sum):
            # if sentences != self.titles[int(sorted_sims[i][0])]:
            topn_sims.append(int(sorted_sims[i][0]))
            sum = sum - 1
            i = i + 1

        topn_queries_num = topn_sims  # 选出的n个问题的编号
        topn_queries = [self.titles[i] for i in topn_queries_num]  # 选出的n个问题
        topn_values = [dict_sims[i] for i in topn_queries_num]  # 选出的n个问题的相似度
        # 得到匹配结果2
        results_2 = {}
        for i in range(n):
            each = [topn_queries_num[i], topn_queries[i], topn_values[i]]
            each_results_2 = {'index': str(topn_queries_num[i] + 1), 'similarity': str(topn_values[i]),
                              'title': topn_queries[i],
                              'confidence': None}
            results_2[i] = each_results_2
        # 汇总结果1和结果2
        results = {'result1': results_1, 'result2': results_2}
        return results


def parse_args():
    path=os.path.abspath ('..')
    file_name = 'v_train.json'
    stoplist_name = 'stop_words.txt'
    # os.path.join(pathconfig.OTHER_DATA_PATH,stoplist_name)
    parser = argparse.ArgumentParser(description='Generate the json file')
    # parser.add_argument('--jsondata_file', help='jsondata_file', default=os.path.join(pathconfig.JSON_DATA_PATH,file_name), type=str)
    parser.add_argument('--jsondata_file', help='jsondata_file',
                        default=r'E:\2018\self-study\panchuang\w-web5\Intelligent_QuaAndAns\Data\json_Data\1\1_2018-08-14-18-43-11_context.json', type=str)
    parser.add_argument('--stop_words_file', help='stop_words_file', default=r'E:\2018\self-study\panchuang\w-web5\Intelligent_QuaAndAns\Data\other_Data', type=str)
    parser.add_argument ('--dictionary_filepath', help='dictionary_filename', default=r'E:\2018\self-study\panchuang\w-web5\Intelligent_QuaAndAns\Data\other_Data\question_dictionary1.dict', type=str)
    args = parser.parse_args()
    return args
import json

if __name__ == '__main__':
    args = parse_args()
    with open(args.jsondata_file, 'r') as f:
        data = json.load(f)
    baseline = Tf_Idf_Model(data, args.dictionary_filepath, args.stop_words_file)

    test_sentence = '请问下物流公司税负率多少'
    results = baseline.get_topn_sims(test_sentence)
    print(results)

