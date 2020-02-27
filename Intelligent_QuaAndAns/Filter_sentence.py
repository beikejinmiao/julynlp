#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : szu-hwj

'''
该文件用来对得到的结果进行筛选
'''

import jieba

def load_target_sentence(results, n=5):
    """
    :param results: 模型返回的结果
    :param n: 用户需要匹配问题的个数
    :return: 得到我们要对其进行过滤的目标句子
    """
    # dict_results = dict(results)
    target = -1  # 我们要进行过滤的目标句子
    target_key = -1  # 目标句子在result2字典中的key，方便后面在result2中找到该句子
    result2 = results['result2']
    for i in range(n):
        each_result_2 = result2[i]
        # 0.65 应该改成0.9 ，现在先检测一下其他
        # result[2]为该匹配问题的相似度
        if float(each_result_2['similarity']) > 0.65 and float(each_result_2['similarity']) < 1.0:
            target = each_result_2
            target_key = i
            break
    if target != -1:
        target_sentence = target['title']   # 匹配问题内容
    else:
        target_sentence = 0   # 0表示匹配的问题没有相似度大于我们预定值的
    return target_sentence, target_key


def get_same_rate(input_sentence, target_sentence, stop_list):
    """
    :param input_sentence: 用户输入的问题
    :param target_sentence: 要过滤的问题
    :return: 检测过滤问题和用户输入问题的字相同率
    """
    input_length = len(input_sentence)
    input_words = []
    for i in input_sentence:
        if i not in stop_list:
            input_words.append(i)
    target_words = []
    for i in target_sentence:
        if i not in stop_list:
            target_words.append(i)
    same_words = 0
    for i in target_words:
        if i in input_words:
            same_words += 1
    same_words_rate = same_words / input_length
    return same_words_rate


def length_difference_rate(input_sentence, target_sentence):
    input_len = len(input_sentence)
    target_sentence = len(target_sentence)
    diff = abs(input_len - target_sentence)
    len_diff_rate = diff / input_len
    return len_diff_rate

def compute_inverse(input_sentence, target_sentence, stop_list):
    """
    计算句子的逆序数
    :param input_sentence: 用户输入问题
    :param target_sentence: 要检测的问题
    :param stop_list: 停用词表
    :return: 逆序数
    """
    # 切词去停用词，停用词不考虑其逆序数
    input_list = [word for word in jieba.cut(input_sentence) if word not in stop_list]
    target_list = [word for word in jieba.cut(target_sentence) if word not in stop_list]
    # 将输入句子的每个词从小到大编号
    keys = list(range(len(input_list)))
    input_dict = {}
    for i in range(len(input_list)):
        input_dict[i] = input_list[i]
    # 生成目标句子的编号序列， 如果目标句子中的词在输入句子中，则该词的编号是输入句子中该词的编号，如果不在则不用计算逆序
    target_index = []
    values_list = list(input_dict.values())
    for i in target_list:
        if i in values_list:
            i_index = values_list.index(i)
            target_index.append(i_index)
    # 计算逆序数
    inverse_number = 0
    for i, index in enumerate(target_index):
        for forward_index in target_index[i + 1:]:
            if forward_index < index:
                inverse_number += 1
    combination_num = len(target_index) * (len(target_index)-1) / 2
    inverse_rate = inverse_number / combination_num
    return inverse_rate