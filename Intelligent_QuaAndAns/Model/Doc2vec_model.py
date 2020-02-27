#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  :
import gensim
import jieba
import os
import json
import time
from gensim.models.doc2vec import Doc2Vec

class Doc2vec_model:
    def __init__(self):
        self.cwd = os.getcwd()
        # with open(self.cwd + r'\X_train.json', 'r') as f:
        with open(os.path.join(self.cwd, 'X_train.json'), 'r') as f:
            data = json.load(f)
            self.titles = []  # 未分词的问题
            self.content = []  # 已分词去停用词的问题
            for i in range(len(data)):
                self.titles.append(data[i]["title"])
                self.content.append(data[i]["split_title"])
        # with open(self.cwd + '\stop_words.txt', encoding='utf-8') as f:
        with open(os.path.join(self.cwd, 'stop_words.txt'), encoding='utf-8') as f:
            lines = f.readlines()
        self.stop_list = [x.strip() for x in lines]
        self.dv_train = self.get_dv_train()

    # 把样本格式转换成doc2vec格式
    def get_dv_train(self):
        TaggededDocument = gensim.models.doc2vec.TaggedDocument
        dv_train = []
        for index, text in enumerate(self.content):
            word_list = text.split(' ')
            document = TaggededDocument(word_list, tags=[index])
            dv_train.append(document)
        return dv_train

    # 问题相似，能区别需要min_count=1，这样最细小的区别就是频率最小的词
    # window也要小一点，可以尝试用dm=0来试下 PV-DBOW 输入是Paragraph vector
    def train(self):
        if os.path.exists('acc_doc2vec.model'):
            model = Doc2Vec.load('acc_doc2vec.model')
        else:
            model = Doc2Vec(self.dv_train, min_count=1, window=3, size=100, sample=1e-3, nagative=5, workers=4, dm=0)
            model.train(self.dv_train, total_examples=model.corpus_count, epochs=100)
            model.save('acc_doc2vec.model')
        return model

    def split_word(self, query):
        words = jieba.cut(query)
        result = ' '.join([word for word in words if word not in self.stop_list])
        return result

    def get_topn_sims(self, sentences, n=5):
        split_sent = self.split_word(sentences)
        results_1 = {'title': sentences, 'split_title': split_sent}
        list_split_sent = split_sent.split()
        dv_model = self.train()
        inferred_vector = dv_model.infer_vector(doc_words=list_split_sent, alpha=0.025, steps=5000)
        sims = dv_model.docvecs.most_similar([inferred_vector], topn=n+1)  # 形如：[(编号，相似度),...]
        # 去除匹配到与原问题一样的结果，认为原问题的相似度一定最高
        top_sim_num = sims[0][0]
        self.titles[top_sim_num] = ''.join(self.titles[top_sim_num].split())
        if sentences == self.titles[top_sim_num]:
            sims = sims[1:]
        else:
            sims = sims[:-1]
        results_2 = []
        for count, sim in sims:
            sim = "%.2f%%" % (sim * 100)  # 转换成百分数形式
            each_results_2 = {'index': str(count), 'similarity': sim,
                              'title': self.titles[count],
                              'confidence': None}
            results_2.append(each_results_2)
        results = {'result1': results_1, 'result2': results_2}
        return results


if __name__ == '__main__':
    start = time.time()
    # test_sentence = '老师请问下，结转成本的金额是怎么算的啊？'
    test_sentence = '请问老师知道e会计云服务怎么结转的吗？'
    d2v = Doc2vec_model()
    results = d2v.get_topn_sims(test_sentence)
    print('spent %.2f times to run' % (time.time() - start))
    print(results)



